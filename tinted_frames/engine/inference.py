"""
Inference Engine Module

Provides unified model management and inference capabilities for VLM analysis.
Supports Qwen2.5-VL with attention tracking and prompt tuning.
"""

import torch
import os
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from tinted_frames.data.samplers import MCQ_POSTFIX, YESNO_POSTFIX, OPEN_ENDED_POSTFIX

class ModelManager:
    """Manages model loading and configuration for VLM inference."""

    def __init__(
        self,
        model_path: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        device_map: str = 'auto',
        torch_dtype: str = 'auto',
        aug_model_path: str = None,
        load_finetune_model_mode: str = 'none',
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """
        Args:
            model_path: Path or name of the pretrained model
            device_map: Device mapping strategy
            torch_dtype: Torch dtype for model weights
            aug_model_path: Path to finetuned checkpoint dir (for ptuning)
            load_finetune_model_mode: 'none' or 'ptuning_{num_soft_tokens}_{position}'
            min_pixels: Minimum pixels for image processing
            max_pixels: Maximum pixels for image processing
        """
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self.aug_model_path = aug_model_path
        self.load_finetune_model_mode = load_finetune_model_mode
        self.special_token_config = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def load_model(self):
        if self.load_finetune_model_mode == 'none':
            return self._load_standard_model()
        assert self.load_finetune_model_mode.startswith('ptuning'), \
            f"Unknown finetune model mode: '{self.load_finetune_model_mode}'"
        return self._load_ptuning_model()

    def _load_standard_model(self):
        """Load model and processor."""
        if self.model is None:
            print(f"Loading model from: {self.model_path}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                attn_implementation="eager",
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )
            self.model.eval()
            print("Model loaded successfully!")
        return self.model, self.processor

    def _load_ptuning_model(self):
        """Load base model with prompt-tuning special token embeddings."""
        if self.model is None:
            print(f"Loading ptuning model: base={self.model_path}, checkpoint={self.aug_model_path}")

            # Parse mode string: ptuning_{num_soft_tokens}_{position}
            parts = self.load_finetune_model_mode.split('_')
            num_soft_tokens = int(parts[1])
            position = parts[2]  # prefix, postfix, infix, or both

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                attn_implementation="eager",
            )

            # Load processor from checkpoint (has special tokens in tokenizer)
            self.processor = AutoProcessor.from_pretrained(
                self.aug_model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )

            # Resize embeddings to accommodate special tokens
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

            # Load learned special embeddings
            emb_path = os.path.join(self.aug_model_path, "special_embeddings.pt")
            special_embeddings = torch.load(emb_path, weights_only=True, map_location='cpu')
            print(f"[ptuning] Loaded special embeddings from {emb_path}, shape={special_embeddings.shape}")

            # Find start ID of special tokens
            start_id = self.processor.tokenizer.convert_tokens_to_ids("<framing_0_0>")
            end_id = start_id + special_embeddings.shape[0]

            # Write learned embeddings into model embed_tokens
            with torch.no_grad():
                self.model.language_model.embed_tokens.weight[start_id:end_id] = special_embeddings.to(
                    self.model.language_model.embed_tokens.weight.dtype
                )
            print(f"[ptuning] Wrote embeddings into token IDs {start_id}:{end_id}")

            # Build special_token_config
            multiplier = 2 if position == "both" else 1
            tokens_per_framing = multiplier * num_soft_tokens

            prefix_texts = []
            postfix_texts = []
            for framing_idx in range(4):
                framing_tokens = [f"<framing_{framing_idx}_{i}>" for i in range(tokens_per_framing)]
                if position == "both":
                    prefix_tokens = framing_tokens[:num_soft_tokens]
                    postfix_tokens = framing_tokens[num_soft_tokens:]
                elif position in ("prefix", "infix"):
                    prefix_tokens = framing_tokens
                    postfix_tokens = []
                else:  # postfix
                    prefix_tokens = []
                    postfix_tokens = framing_tokens
                prefix_texts.append(" ".join(prefix_tokens))
                postfix_texts.append(" ".join(postfix_tokens))

            self.special_token_config = {
                "position": position,
                "prefix_texts": prefix_texts,
                "postfix_texts": postfix_texts,
            }

            self.model.eval()
            print("Ptuning model loaded successfully!")

        return self.model, self.processor

    def get_model(self):
        """Get loaded model (loads if not already loaded)."""
        if self.model is None:
            self.load_model()
        return self.model

    def get_processor(self):
        """Get processor (loads if not already loaded)."""
        if self.processor is None:
            self.load_model()
        return self.processor


class InferenceEngine:
    """
    Unified inference engine for VLM attention analysis.

    Handles input preparation, model inference with attention tracking,
    output processing, and attention steering via hooks.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        max_new_tokens: int = 32,
        output_attentions: bool = True
    ):
        self.model_manager = model_manager
        self.max_new_tokens = max_new_tokens
        self.output_attentions = output_attentions

    @staticmethod
    def option_to_answer(selected_option: str, options: list, task_type: str) -> str:
        """Convert raw option text from answer ranking back to the expected answer format.

        For MCQ: returns the letter (A, B, C, D) corresponding to the option's position.
        For yes/no: returns the option text as-is.
        For open_ended or empty options: returns the option text as-is.
        """
        if not options or task_type in ('open_form', 'open_ended'):
            return selected_option
        if task_type in ('yes_no', 'yesno'):
            return selected_option
        try:
            idx = options.index(selected_option)
            return chr(65 + idx)
        except ValueError:
            return selected_option

    def prepare_inputs(
        self,
        image: Image.Image,
        question_text: str,
        framing_idx: int = -1,
        use_answer_ranking: bool = False
    ) -> Tuple[Any, Dict]:
        """
        Prepare inputs for the model with image and question.

        Args:
            image: PIL Image
            question_text: Question string
            framing_idx: Framing index for special token injection (0-3)
            use_answer_ranking: Whether to use answer ranking mode

        Returns:
            inputs: Model inputs
            messages: Message format for processor
        """
        processor = self.model_manager.get_processor()

        # Inject special tokens if prompt-tuning config exists
        if self.model_manager.special_token_config is not None and framing_idx in range(4):
            config = self.model_manager.special_token_config
            prefix_text = config["prefix_texts"][framing_idx]
            postfix_text = config["postfix_texts"][framing_idx]
            if config["position"] in ("prefix", "both") and prefix_text:
                question_text = prefix_text + " " + question_text + " "
            if config["position"] == "infix" and prefix_text:
                if MCQ_POSTFIX in question_text:
                    question_text = question_text.replace(MCQ_POSTFIX, "") + " " + prefix_text + " " + MCQ_POSTFIX
                elif YESNO_POSTFIX in question_text:
                    question_text = question_text.replace(YESNO_POSTFIX, "") + " " + prefix_text + " " + YESNO_POSTFIX
                elif OPEN_ENDED_POSTFIX in question_text:
                    question_text = question_text.replace(OPEN_ENDED_POSTFIX, "") + " " + prefix_text + " " + OPEN_ENDED_POSTFIX
                else:
                    question_text = question_text + " " + prefix_text

            if config["position"] in ("postfix", "both") and postfix_text:
                question_text = question_text + " " + postfix_text

        if use_answer_ranking:
            if MCQ_POSTFIX in question_text:
                question_text = question_text.replace(MCQ_POSTFIX, "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        return inputs, messages

    def run_inference_answer_ranking(
        self,
        model,
        processor,
        inputs,
        options: List[str],
        img_start_idx: int = -1,
        img_end_idx: int = -1,
        postfix_start_idx: int = -1,
        input_sequence_mapper: list = None,
    ) -> Dict[str, Any]:
        """Score each answer option by cross-entropy loss (answer ranking)."""
        input_ids = inputs.input_ids
        device = input_ids.device
        best_score = float('inf')
        best_option = options[0]

        with torch.no_grad():
            for option in options:
                option_ids = processor.tokenizer.encode(option, add_special_tokens=False)
                option_ids = torch.tensor([option_ids], device=device)
                option_len = option_ids.shape[1]

                combined_ids = torch.cat([input_ids, option_ids], dim=1)

                forward_kwargs = {}
                if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                    extended_mask = torch.ones(1, option_len, device=device, dtype=inputs.attention_mask.dtype)
                    forward_kwargs['attention_mask'] = torch.cat([inputs.attention_mask, extended_mask], dim=1)
                if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                    forward_kwargs['pixel_values'] = inputs.pixel_values
                if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                    forward_kwargs['image_grid_thw'] = inputs.image_grid_thw
                if hasattr(inputs, 'pixel_values_videos') and inputs.pixel_values_videos is not None:
                    forward_kwargs['pixel_values_videos'] = inputs.pixel_values_videos
                if hasattr(inputs, 'video_grid_thw') and inputs.video_grid_thw is not None:
                    forward_kwargs['video_grid_thw'] = inputs.video_grid_thw

                labels = combined_ids.clone()
                labels[:, :-option_len] = -100
                outputs = model(input_ids=combined_ids, labels=labels, **forward_kwargs)
                loss = outputs.loss.item()

                if loss < best_score:
                    best_score = loss
                    best_option = option

        input_len = input_ids.shape[1]
        result = {
            'gen_dict': None,
            'inputs': inputs,
            'output_text': best_option,
            'img_start_idx': img_start_idx,
            'img_end_idx': img_end_idx,
            'input_len': input_len,
            'postfix_start_idx': postfix_start_idx,
            'input_sequence_mapper': input_sequence_mapper,
        }

        return result

    def run_inference(
        self,
        image: Image.Image,
        question_text: str,
        framing_idx: int = -1,
        use_answer_ranking: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Run inference and return generation outputs with attention maps.

        Args:
            image: PIL Image
            question_text: Question string
            framing_idx: Framing index for special token injection (0-3)
            use_answer_ranking: List of answer options for ranking mode

        Returns:
            Dictionary with gen_dict, inputs, output_text, token indices, etc.
        """
        model = self.model_manager.get_model()
        processor = self.model_manager.get_processor()

        inputs, messages = self.prepare_inputs(image, question_text, framing_idx=framing_idx, use_answer_ranking=use_answer_ranking)

        # Find image token positions (Qwen2.5-VL specific token IDs)
        IMAGE_TOKEN_IDX = 151655
        POSTFIX_TOKEN_IDX = 5501

        try:
            img_start_idx = inputs.input_ids[0].tolist().index(IMAGE_TOKEN_IDX)
            img_end_idx = img_start_idx + (inputs.input_ids[0] == IMAGE_TOKEN_IDX).sum().item()
        except ValueError:
            img_start_idx, img_end_idx = -1, -1

        image_token_indices = [i for i, token in enumerate(inputs.input_ids[0].tolist()) if token == 151645]
        first_im_end_idx = image_token_indices[0] if len(image_token_indices) > 0 else -1
        second_im_end_idx = image_token_indices[1] if len(image_token_indices) > 1 else -1

        try:
            postfix_start_idx = inputs.input_ids[0].tolist().index(POSTFIX_TOKEN_IDX)
        except ValueError:
            postfix_start_idx = -1

        input_sequence_mapper = [
            (0, 3, 'special'),
            (3, first_im_end_idx, 'system'),
            (first_im_end_idx, img_start_idx, 'special'),
            (img_start_idx, img_end_idx, 'image'),
            (img_end_idx, img_end_idx + 1, 'special'),
            (img_end_idx + 1, postfix_start_idx, 'question'),
            (postfix_start_idx, second_im_end_idx, 'postfix'),
            (second_im_end_idx, len(inputs.input_ids[0]), 'special'),
        ]

        # Answer ranking mode
        if use_answer_ranking:
            return self.run_inference_answer_ranking(
                model, processor, inputs, use_answer_ranking,
                img_start_idx=img_start_idx, img_end_idx=img_end_idx,
                postfix_start_idx=postfix_start_idx,
                input_sequence_mapper=input_sequence_mapper,
            )

        # Generate with attention output
        with torch.no_grad():
        
            gen_dict = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                output_attentions=self.output_attentions,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        input_len = inputs.input_ids[0].shape[0]
        output_text = processor.batch_decode(
            gen_dict.sequences[:, input_len:],
            skip_special_tokens=True
        )[0]

        result = {
            'gen_dict': gen_dict,
            'inputs': inputs,
            'output_text': output_text,
            'img_start_idx': img_start_idx,
            'img_end_idx': img_end_idx,
            'input_len': input_len,
            'postfix_start_idx': postfix_start_idx,
            'input_sequence_mapper': input_sequence_mapper
        }

        return result

    def run_inference_from_path(
        self,
        image_path: str,
        question: str,
        max_new_tokens: Optional[int] = None,
        framing_idx: int = -1,
        use_answer_ranking: List[str] = [],
    ) -> Tuple:
        """Simplified interface for job scripts - loads image from path."""
        image = Image.open(image_path).convert('RGB')

        original_max_tokens = self.max_new_tokens
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens

        result = self.run_inference(
            image, question,
            framing_idx=framing_idx,
            use_answer_ranking=use_answer_ranking
        )

        self.max_new_tokens = original_max_tokens

        token_indices = {
            'img_start_idx': result['img_start_idx'],
            'img_end_idx': result['img_end_idx'],
            'input_len': result['input_len'],
            'output_token_ids': result['gen_dict'].sequences[0, result['input_len']:].tolist() if result['gen_dict'] is not None else [],
            'postfix_start_idx': result['postfix_start_idx'],
            'input_sequence_mapper': result['input_sequence_mapper']
        }

        return_tuple = [result['gen_dict'], result['inputs'], result['output_text'], token_indices]

        return tuple(return_tuple)