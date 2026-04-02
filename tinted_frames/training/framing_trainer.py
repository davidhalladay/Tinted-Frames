"""
Framing Consistency Trainer

Subclasses transformers.Trainer and overrides compute_loss to add:
  - CE loss per framing (open, yes/no, MCQ) on answer tokens only
  - Attention rollout KL (default) or MSE to align visual grounding across framings

PEFT is applied externally before the trainer is created.
Multi-GPU DDP / DeepSpeed are handled transparently by the Trainer base class.
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from ..engine.attention import AttentionRollout
from ..engine.model_config import get_embed_layer


# ---------------------------------------------------------------------------
# Attention extraction (gradient-compatible)
# ---------------------------------------------------------------------------

def extract_rollout_attention(
    attentions: Tuple,
    answer_starts: List[int],
    answer_ends: List[int],
    img_starts: List[int],
    img_ends: List[int],
    layer_start: int = 10,
    layer_end: int = 24,
    rollout_obj: Optional[AttentionRollout] = None,
) -> torch.Tensor:
    """
    Gradient-compatible attention rollout from answer tokens -> image patches.

    Returns:
        [batch, heads, n_img] — normalized probability maps.
    """
    if rollout_obj is None:
        rollout_obj = AttentionRollout(
            residual_weight=0.5, start_layer=layer_start, end_layer=layer_end
        )

    selected_layers = list(attentions[layer_start:layer_end])
    batch_size = selected_layers[0].shape[0]
    num_heads = selected_layers[0].shape[1]

    batch_maps = []
    for b in range(batch_size):
        per_sample = [layer[b] for layer in selected_layers]
        rollout, rollout_all_layers = rollout_obj.compute_rollout(per_sample)

        a_s, a_e = answer_starts[b], answer_ends[b]
        i_s, i_e = img_starts[b], img_ends[b]
        n_img = i_e - i_s

        if n_img <= 0 or a_e <= a_s:
            batch_maps.append(rollout.new_zeros(num_heads, max(n_img, 1)))
            continue

        attn_avg = rollout[:, a_s - 1 : a_e, i_s:i_e].mean(dim=1)
        batch_maps.append(attn_avg)

    min_img = min(m.shape[-1] for m in batch_maps)
    return torch.stack([m[..., :min_img] for m in batch_maps], dim=0)


# ---------------------------------------------------------------------------
# CE loss on answer tokens only
# ---------------------------------------------------------------------------

def compute_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_starts: List[int],
    answer_ends: List[int],
) -> torch.Tensor:
    """Compute cross-entropy loss on answer tokens only."""
    input_ids = input_ids.to(logits.device)
    total_loss = logits.new_zeros(())
    total_tokens = 0

    for b in range(logits.shape[0]):
        a_s, a_e = answer_starts[b], answer_ends[b]
        if a_e <= a_s:
            continue
        loss = F.cross_entropy(
            logits[b, a_s - 1 : a_e - 1], input_ids[b, a_s:a_e], reduction="sum"
        )
        total_loss = total_loss + loss
        total_tokens += a_e - a_s

    if total_tokens == 0:
        return logits.new_zeros((), requires_grad=True)
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class FramingTrainer(Trainer):
    """
    transformers.Trainer subclass that computes:

        L = CE(open) + CE(open_constrained) + CE(yesno) + CE(mcq)
          + lambda_mass  * MSE( am_open_c.sum(-1), am_other.sum(-1) )
          + lambda_shape * KL ( norm(am_open_c) || norm(am_other) )

    am_open_c (open-constrained) is detached — used as the grounding reference
    because its answers are short and produce tighter attention maps.

    Extra constructor args:
      lambda_mass  (float) : weight for image-attention mass MSE   [default: 5.0]
      lambda_shape (float) : weight for patch-distribution KL      [default: 0.1]
      layer_range  (tuple) : (start, end) layer slice              [default: (10, 24)]
    """

    def __init__(
        self,
        *args,
        lambda_mass: float = 5.0,
        lambda_shape: float = 0.1,
        layer_range: Tuple[int, int] = (10, 24),
        processor=None,
        model_config: Optional[Dict[str, Any]] = None,
        weighting_scheme: str = "ema_confidence",
        desink: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_mass = lambda_mass
        self.lambda_shape = lambda_shape
        self.layer_start, self.layer_end = layer_range
        self.processor = processor
        self.model_config = model_config
        self.weighting_scheme = weighting_scheme
        self.desink = desink
        self.rollout = AttentionRollout(
            residual_weight=0.5,
            start_layer=self.layer_start,
            end_layer=self.layer_end,
        )
        # EMA confidence tracking
        self.conf_ema_mean = 0.5
        self.conf_ema_std = 0.25
        self.ema_decay = 0.95
        self.ema_norm_conf_correct = 0.5
        self.ema_norm_conf_incorrect = 0.5

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _desink(self, attn_map: torch.Tensor) -> torch.Tensor:
        """Zero out top 2% tokens by attention value per sample per head."""
        k = max(1, int(attn_map.shape[-1] * 0.02))
        topk_vals, _ = attn_map.topk(k, dim=-1)
        threshold = topk_vals[..., -1:]
        mask = attn_map < threshold
        return attn_map * mask

    def _unwrap(self, model):
        """Unwrap DDP/FSDP to get the base model."""
        if hasattr(model, "module"):
            return model.module
        return model

    def _forward_framing(self, model, framing_batch: Dict[str, Any], framing_idx: int):
        """
        Forward one framing through the model with output_attentions=True.
        Returns (logits, attentions).
        """
        device = next(model.parameters()).device
        input_ids = framing_batch["input_ids"].to(device)
        attention_mask = framing_batch["attention_mask"].to(device)

        model_kwargs: Dict[str, Any] = {}
        if "pixel_values" in framing_batch:
            pv = framing_batch["pixel_values"]
            if isinstance(pv, list):
                model_kwargs["pixel_values"] = torch.cat(
                    [
                        p.to(device).unsqueeze(0) if p.dim() == 3 else p.to(device)
                        for p in pv
                    ],
                    dim=0,
                )
            else:
                model_kwargs["pixel_values"] = pv.to(device)

        if "image_grid_thw" in framing_batch:
            model_kwargs["image_grid_thw"] = framing_batch["image_grid_thw"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            **model_kwargs,
        )
        return outputs.logits, outputs.attentions

    # ------------------------------------------------------------------
    # Core: custom loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        inputs is the quadruplet batch from framing_collate_fn:
            {"open": {...}, "open_constrained": {...}, "yesno": {...}, "mcq": {...}}
        """
        open_b = inputs["open"]
        open_c_b = inputs["open_constrained"]
        yesno_b = inputs["yesno"]
        mcq_b = inputs["mcq"]

        # --- four forward passes ---
        open_logits, open_attn = self._forward_framing(model, open_b, 0)
        yesno_logits, yesno_attn = self._forward_framing(model, yesno_b, 1)
        mcq_logits, mcq_attn = self._forward_framing(model, mcq_b, 2)
        open_c_logits, open_c_attn = self._forward_framing(model, open_c_b, 3)

        # --- CE losses ---
        ce_open = compute_ce_loss(
            open_logits, open_b["input_ids"], open_b["answer_starts"], open_b["answer_ends"]
        )
        ce_yesno = compute_ce_loss(
            yesno_logits, yesno_b["input_ids"], yesno_b["answer_starts"], yesno_b["answer_ends"]
        )
        ce_mcq = compute_ce_loss(
            mcq_logits, mcq_b["input_ids"], mcq_b["answer_starts"], mcq_b["answer_ends"]
        )
        ce_open_c = compute_ce_loss(
            open_c_logits, open_c_b["input_ids"], open_c_b["answer_starts"], open_c_b["answer_ends"]
        )

        # --- attention rollout ---
        def _attn(fb, attn):
            return extract_rollout_attention(
                attentions=attn,
                answer_starts=fb["answer_starts"],
                answer_ends=fb["answer_ends"],
                img_starts=fb["img_starts"],
                img_ends=fb["img_ends"],
                layer_start=self.layer_start,
                layer_end=self.layer_end,
                rollout_obj=self.rollout,
            )

        am_open = _attn(open_b, open_attn)
        am_yesno = _attn(yesno_b, yesno_attn)
        am_mcq = _attn(mcq_b, mcq_attn)
        am_open_c = _attn(open_c_b, open_c_attn)

        if self.desink:
            am_open_c = self._desink(am_open_c)
            am_yesno = self._desink(am_yesno)
            am_mcq = self._desink(am_mcq)

        # --- Correctness weighting ---
        batch_size = open_c_logits.shape[0]
        correctness_weights = self._compute_correctness_weights(
            batch_size, open_c_logits, open_c_b, yesno_logits, yesno_b, mcq_logits, mcq_b
        )

        print(f"Correctness weights: {correctness_weights}")

        correctness_weights = torch.tensor(
            correctness_weights, device=am_open_c.device, dtype=am_open_c.dtype
        ).view(batch_size, 1, 1)

        # --- Mass & Shape losses ---
        n = min(am_open_c.shape[-1], am_yesno.shape[-1], am_mcq.shape[-1])

        target_mass = am_open_c[..., :n].sum(-1).detach()
        mass_yesno = F.mse_loss(am_yesno[..., :n].sum(-1), target_mass, reduction="none")
        mass_mcq = F.mse_loss(am_mcq[..., :n].sum(-1), target_mass, reduction="none")
        mass_loss = (mass_yesno * correctness_weights).mean() + (
            mass_mcq * correctness_weights
        ).mean()

        def _norm(x):
            return x / x.sum(-1, keepdim=True).clamp(min=1e-8)

        target_shape = _norm(am_open_c[..., :n]).detach()
        shape_yesno = F.kl_div(
            _norm(am_yesno[..., :n]).clamp(min=1e-8).log(), target_shape, reduction="none"
        ).sum(dim=-1)
        shape_mcq = F.kl_div(
            _norm(am_mcq[..., :n]).clamp(min=1e-8).log(), target_shape, reduction="none"
        ).sum(dim=-1)
        shape_loss = (shape_yesno * correctness_weights).mean() + (
            shape_mcq * correctness_weights
        ).mean()

        loss = (
            ce_yesno
            + ce_mcq
            + self.lambda_mass * mass_loss
            + self.lambda_shape * shape_loss
        )

        # --- Logging ---
        if self.model.training:
            avg_weight = correctness_weights.mean().item()
            num_correct = (correctness_weights.squeeze() > 0.5).sum().item()
            log_dict = {
                "loss/ce_open": ce_open.item(),
                "loss/ce_open_constrained": ce_open_c.item(),
                "loss/ce_yesno": ce_yesno.item(),
                "loss/ce_mcq": ce_mcq.item(),
                "loss/attn_mass": (self.lambda_mass * mass_loss).item(),
                "loss/attn_shape": (self.lambda_shape * shape_loss).item(),
                "attn/mean_open": am_open.sum(dim=-1).mean().item(),
                "attn/mean_open_constrained": am_open_c.sum(dim=-1).mean().item(),
                "attn/mean_yesno": am_yesno.sum(dim=-1).mean().item(),
                "attn/mean_mcq": am_mcq.sum(dim=-1).mean().item(),
                "correctness/avg_weight": avg_weight,
                "correctness/num_correct": num_correct,
                "correctness/accuracy": num_correct / batch_size,
            }
            if self.weighting_scheme == "ema_confidence":
                log_dict.update(
                    {
                        "ema/norm_conf_correct": self.ema_norm_conf_correct,
                        "ema/norm_conf_incorrect": self.ema_norm_conf_incorrect,
                        "ema/running_mean": self.conf_ema_mean,
                        "ema/running_std": self.conf_ema_std,
                    }
                )
            self.log(log_dict)

        if return_outputs:
            return loss, (open_logits, open_c_logits, yesno_logits, mcq_logits)
        return loss

    # ------------------------------------------------------------------
    # Correctness weighting strategies
    # ------------------------------------------------------------------

    def _compute_correctness_weights(
        self, batch_size, open_c_logits, open_c_b, yesno_logits, yesno_b, mcq_logits, mcq_b
    ) -> List[float]:
        if self.weighting_scheme == "confidence":
            return self._weights_confidence(batch_size, open_c_logits, open_c_b)
        elif self.weighting_scheme == "ema_confidence":
            return self._weights_ema_confidence(batch_size, open_c_logits, open_c_b)
        else:
            return self._weights_binary(
                batch_size, open_c_logits, open_c_b, yesno_logits, yesno_b, mcq_logits, mcq_b
            )

    def _weights_confidence(self, batch_size, open_c_logits, open_c_b) -> List[float]:
        weights = []
        for b in range(batch_size):
            a_s, a_e = open_c_b["answer_starts"][b], open_c_b["answer_ends"][b]
            if a_e <= a_s:
                weights.append(0.0)
                continue
            probs = torch.softmax(open_c_logits[b, a_s - 1 : a_e - 1], dim=-1)
            gt_ids = open_c_b["input_ids"][b, a_s:a_e].to(probs.device)
            gt_probs = probs[torch.arange(len(gt_ids), device=probs.device), gt_ids]
            weights.append(gt_probs.mean().item())
        return weights

    def _weights_ema_confidence(self, batch_size, open_c_logits, open_c_b) -> List[float]:
        weights = []
        for b in range(batch_size):
            a_s, a_e = open_c_b["answer_starts"][b], open_c_b["answer_ends"][b]
            if a_e <= a_s:
                weights.append(0.0)
                continue
            probs = torch.softmax(open_c_logits[b, a_s - 1 : a_e - 1], dim=-1)
            gt_ids = open_c_b["input_ids"][b, a_s:a_e].to(probs.device)
            gt_probs = probs[torch.arange(len(gt_ids), device=probs.device), gt_ids]
            confidence = gt_probs.mean().item()

            z = (confidence - self.conf_ema_mean) / (self.conf_ema_std + 1e-8)
            norm_conf = torch.sigmoid(torch.tensor(z)).item()

            pred_ids = open_c_logits[b, a_s - 1 : a_e - 1].argmax(dim=-1)
            pred_text = self.processor.decode(pred_ids, skip_special_tokens=True).strip().lower()
            gt_text = self.processor.decode(gt_ids, skip_special_tokens=True).strip().lower()
            is_correct = (gt_text in pred_text or pred_text in gt_text) and pred_text != ""

            weight = norm_conf if is_correct else norm_conf * 0.5
            weights.append(weight)

            if is_correct:
                self.ema_norm_conf_correct = (
                    self.ema_decay * self.ema_norm_conf_correct + (1 - self.ema_decay) * norm_conf
                )
            else:
                self.ema_norm_conf_incorrect = (
                    self.ema_decay * self.ema_norm_conf_incorrect
                    + (1 - self.ema_decay) * norm_conf
                )
            self.conf_ema_mean = (
                self.ema_decay * self.conf_ema_mean + (1 - self.ema_decay) * confidence
            )
            self.conf_ema_std = (
                self.ema_decay * self.conf_ema_std
                + (1 - self.ema_decay) * abs(confidence - self.conf_ema_mean)
            )
        return weights

    def _weights_binary(
        self, batch_size, open_c_logits, open_c_b, yesno_logits, yesno_b, mcq_logits, mcq_b
    ) -> List[float]:
        weights = []
        for b in range(batch_size):
            a_s, a_e = open_c_b["answer_starts"][b], open_c_b["answer_ends"][b]
            pred_ids = open_c_logits[b, a_s - 1 : a_e - 1].argmax(dim=-1)
            gt_ids = open_c_b["input_ids"][b, a_s:a_e]

            pred_text = self.processor.decode(pred_ids, skip_special_tokens=True).strip().lower()
            gt_text = self.processor.decode(gt_ids, skip_special_tokens=True).strip().lower()

            is_correct = (gt_text in pred_text or pred_text in gt_text) and pred_text != ""
            weights.append(1.0 if is_correct else 0.2)
        return weights

    # ------------------------------------------------------------------
    # Save: prompt-tuning checkpoints store only special_embeddings.pt
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model, trial):
        base = self._unwrap(model)
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_folder = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")

        if hasattr(base, "special_embedding_module"):
            self._skip_model_save = True
            try:
                super(FramingTrainer, self)._save_checkpoint(model, trial)
            finally:
                self._skip_model_save = False

            emb_path = os.path.join(checkpoint_folder, "special_embeddings.pt")
            torch.save(base.special_embedding_module.embeddings.data.cpu(), emb_path)
            print(f"[FramingTrainer] Saved special embeddings -> {emb_path}")
            if self.processor is not None:
                self.processor.save_pretrained(checkpoint_folder)
        else:
            super(FramingTrainer, self)._save_checkpoint(model, trial)
            if self.processor is not None:
                self.processor.save_pretrained(checkpoint_folder)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self, "_skip_model_save", False):
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            return

        base = self._unwrap(self.model)
        if hasattr(base, "special_embedding_module"):
            with torch.no_grad():
                start_id = base.special_token_start_id
                end_id = base.special_token_end_id
                if self.model_config is not None:
                    embed_layer = get_embed_layer(base, self.model_config)
                else:
                    embed_layer = base.model.language_model.embed_tokens
                embed_layer.weight[start_id:end_id].copy_(
                    base.special_embedding_module.embeddings
                )

        super(FramingTrainer, self)._save(output_dir, state_dict)
        if self.processor is not None and output_dir is not None:
            self.processor.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = model if model is not None else self.model
        base = self._unwrap(target)

        if hasattr(base, "special_embedding_module"):
            emb_path = os.path.join(resume_from_checkpoint, "special_embeddings.pt")
            if not os.path.exists(emb_path):
                raise ValueError(
                    f"Resuming prompt-tuning but no special_embeddings.pt in {resume_from_checkpoint}"
                )
            saved_emb = torch.load(emb_path, weights_only=True, map_location="cpu")
            with torch.no_grad():
                base.special_embedding_module.embeddings.data.copy_(
                    saved_emb.to(base.special_embedding_module.embeddings.dtype)
                )
            print(
                f"[FramingTrainer] Resumed special embeddings "
                f"({saved_emb.numel():,} values) from {emb_path}"
            )
            return

        super(FramingTrainer, self)._load_from_checkpoint(resume_from_checkpoint, model)
