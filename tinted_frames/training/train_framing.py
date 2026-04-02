"""
Framing Consistency Fine-tuning — Entry Point

Single GPU:
    python -m tinted_frames.training.train_framing [args...]

Multi-GPU via torchrun:
    torchrun --nproc_per_node=4 -m tinted_frames.training.train_framing [args...]
"""

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .framing_dataset import FramingDataset, framing_collate_fn
from .framing_trainer import FramingTrainer
from ..engine.model_config import (
    get_model_config,
    get_lora_target_modules,
    get_embed_layer,
)

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_KEY = "qwen2.5vl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Framing consistency fine-tuning")

    # Model
    p.add_argument("--model_path", default=MODEL_PATH, help="HF model path or local checkpoint")

    # Data
    p.add_argument("--data_path", required=True, help="Path to llava_v1_5_mix665k.json")
    p.add_argument("--image_root", required=True, help="Root for training images")
    p.add_argument("--question_mapper_path", default="data/question_mapper.json")
    p.add_argument("--sample_n", type=int, default=0, help="0 = use all 665k")

    # LLM backend for framing generation
    p.add_argument("--llm_backend", choices=["gpt", "vllm"], default="vllm")
    p.add_argument("--gpt_model", default="gpt-4o-mini")
    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--vllm_host", default="localhost")
    p.add_argument("--vllm_port", type=int, default=8000)
    p.add_argument("--vllm_model", default="Qwen3-14B")

    # PEFT
    p.add_argument("--training_mode", choices=["lora", "prompt_tuning"], default="lora")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_layers", type=int, nargs=2, default=[10, 24],
                   metavar=("START", "END"))
    p.add_argument("--num_soft_tokens", type=int, default=16)
    p.add_argument("--prompt_position", choices=["prefix", "postfix", "both", "infix"],
                   default="prefix")

    # Loss
    p.add_argument("--lambda_mass", type=float, default=5.0)
    p.add_argument("--lambda_shape", type=float, default=0.1)
    p.add_argument("--layer_range", type=int, nargs=2, default=[10, 24],
                   metavar=("START", "END"))
    p.add_argument("--weighting_scheme", choices=["binary", "confidence", "ema_confidence"],
                   default="ema_confidence")
    p.add_argument("--desink", action="store_true")

    # Training
    p.add_argument("--output_dir", default="checkpoints/framing_exp_001")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--run_name", default=None)
    p.add_argument("--deepspeed", default=None)

    # Dataset
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--image_size", type=int, default=448, help="0 = no resize")
    p.add_argument("--seed", type=int, default=42)

    # Visualization callback
    p.add_argument("--viz_log_steps", type=int, default=10)
    p.add_argument("--viz_n_samples", type=int, default=5)

    # Ablations
    p.add_argument("--no_mcq_postfix", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, bf16: bool, gradient_checkpointing: bool):
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


# ---------------------------------------------------------------------------
# PEFT setup
# ---------------------------------------------------------------------------

def apply_lora(model, lora_r: int, lora_alpha: int, layer_start: int, layer_end: int,
               model_config: dict):
    from peft import LoraConfig, get_peft_model

    target_modules = get_lora_target_modules(model_config, layer_start, layer_end)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def apply_prompt_tuning(model, processor, num_soft_tokens: int, hidden_size: int,
                        prompt_position: str, model_config: dict):
    """
    Freeze model params and add special framing tokens to the tokenizer.
    Returns a special_token_config dict for the dataset.
    """
    multiplier = 2 if prompt_position == "both" else 1
    tokens_per_framing = multiplier * num_soft_tokens

    special_tokens = []
    for framing_idx in range(4):
        for token_idx in range(tokens_per_framing):
            special_tokens.append(f"<framing_{framing_idx}_{token_idx}>")

    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    print(f"[PromptTuning] Added {num_added} special tokens")

    model.resize_token_embeddings(len(processor.tokenizer))

    prefix_texts = []
    postfix_texts = []
    for framing_idx in range(4):
        framing_tokens = [
            f"<framing_{framing_idx}_{i}>" for i in range(tokens_per_framing)
        ]
        if prompt_position == "both":
            prefix_tokens = framing_tokens[:num_soft_tokens]
            postfix_tokens = framing_tokens[num_soft_tokens:]
        elif prompt_position == "prefix":
            prefix_tokens = framing_tokens
            postfix_tokens = []
        elif prompt_position == "postfix":
            prefix_tokens = []
            postfix_tokens = framing_tokens
        else:  # infix
            prefix_tokens = framing_tokens
            postfix_tokens = []

        prefix_texts.append(" ".join(prefix_tokens))
        postfix_texts.append(" ".join(postfix_tokens))

    framing_token_ids = []
    for framing_idx in range(4):
        framing_tokens = [
            f"<framing_{framing_idx}_{i}>" for i in range(tokens_per_framing)
        ]
        token_ids = [
            processor.tokenizer.convert_tokens_to_ids(t) for t in framing_tokens
        ]
        framing_token_ids.append(token_ids)

    # Initialize from meaningful text
    embed_layer = get_embed_layer(model, model_config)
    with torch.no_grad():
        init_text = "Look at the image carefully when answering the question."
        tokens = processor.tokenizer(init_text, add_special_tokens=False)["input_ids"]
        for framing_idx, token_ids in enumerate(framing_token_ids):
            for i, token_id in enumerate(token_ids):
                init_token_idx = tokens[i % len(tokens)]
                init_embed = embed_layer.weight[init_token_idx].clone()
                init_embed += torch.randn_like(init_embed) * 0.01
                embed_layer.weight[token_id] = init_embed

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    class SpecialEmbeddings(torch.nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.embeddings = embeddings

    special_token_embeddings = torch.nn.Parameter(
        embed_layer.weight[framing_token_ids[0][0] : framing_token_ids[3][-1] + 1].clone()
    )
    model.special_embedding_module = SpecialEmbeddings(special_token_embeddings)
    model.special_token_start_id = framing_token_ids[0][0]
    model.special_token_end_id = framing_token_ids[3][-1] + 1

    def embedding_hook(module, input, output):
        input_ids = input[0]
        embeddings = output
        mask = (input_ids >= model.special_token_start_id) & (
            input_ids < model.special_token_end_id
        )
        if mask.any():
            special_indices = input_ids[mask] - model.special_token_start_id
            embeddings = embeddings.clone()
            embeddings[mask] = model.special_embedding_module.embeddings[special_indices]
        return embeddings

    embed_layer.register_forward_hook(embedding_hook)

    trainable = special_token_embeddings.numel()
    print(
        f"[PromptTuning] Learnable params: {trainable:,} "
        f"({num_added} tokens x {hidden_size} dim)"
    )

    return {
        "position": prompt_position,
        "prefix_texts": prefix_texts,
        "postfix_texts": postfix_texts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_config = get_model_config(MODEL_KEY)

    # ---- Processor ----
    processor = AutoProcessor.from_pretrained(args.model_path)

    # ---- Model ----
    model = load_model(args.model_path, args.bf16, args.gradient_checkpointing)

    # ---- PEFT setup ----
    special_token_config = None
    if args.training_mode == "lora":
        model = apply_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            layer_start=args.lora_target_layers[0],
            layer_end=args.lora_target_layers[1],
            model_config=model_config,
        )
    else:
        special_token_config = apply_prompt_tuning(
            model,
            processor=processor,
            num_soft_tokens=args.num_soft_tokens,
            hidden_size=model.config.hidden_size,
            prompt_position=args.prompt_position,
            model_config=model_config,
        )

    # ---- Dataset ----
    image_size = args.image_size if args.image_size > 0 else None
    dataset = FramingDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        processor=processor,
        question_mapper_path=args.question_mapper_path,
        llm_backend=args.llm_backend,
        gpt_model=args.gpt_model,
        openai_api_key=args.openai_api_key,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        vllm_model=args.vllm_model,
        sample_n=args.sample_n,
        max_seq_len=args.max_seq_len,
        image_size=image_size,
        seed=args.seed,
        special_token_config=special_token_config,
        model_config=model_config,
        no_mcq_postfix=args.no_mcq_postfix,
    )

    # ---- TrainingArguments ----
    run_name = args.run_name or f"framing_{args.training_mode}"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=run_name,
        logging_dir=os.path.join(args.output_dir, "tb_logs"),
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        seed=args.seed,
    )

    # ---- Trainer ----
    trainer = FramingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=framing_collate_fn,
        lambda_mass=args.lambda_mass,
        lambda_shape=args.lambda_shape,
        layer_range=tuple(args.layer_range),
        processor=processor,
        model_config=model_config,
        weighting_scheme=args.weighting_scheme,
        desink=args.desink,
    )

    # ---- Visualization callback ----
    if args.viz_log_steps > 0 and args.viz_n_samples > 0:
        from .framing_viz_callback import FramingVizCallback

        viz_cb = FramingVizCallback(
            dataset=dataset,
            processor=processor,
            viz_log_steps=args.viz_log_steps,
            viz_n_samples=args.viz_n_samples,
            layer_start=args.layer_range[0],
            layer_end=args.layer_range[1],
            logging_dir=os.path.join(args.output_dir, "tb_logs"),
            image_size=image_size,
        )
        trainer.add_callback(viz_cb)

    # Resume from checkpoint if one exists
    last_ckpt = (
        get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    )
    if last_ckpt:
        print(f"[train_framing] Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[train_framing] Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
