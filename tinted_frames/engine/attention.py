"""
Attention Collection and Processing Module

This module provides utilities for collecting and processing attention maps from VLMs.
It supports multiple aggregation strategies and attention rollout computation.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class AttentionRollout:
    """
    Implements Attention Rollout algorithm for Qwen2.5-VL,
    handling KV-cache stitching and causal mask normalization.
    """

    def __init__(self, residual_weight: float = 0.5, start_layer: int = 18, end_layer: int = 24):
        self.residual_weight = residual_weight
        self.start_layer = start_layer
        self.end_layer = end_layer

    def _stitch_attention_matrices(self, all_step_attentions: tuple) -> List[torch.Tensor]:
        """
        Reconstructs the full N x N attention matrix for each layer by stitching
        the prefill attention (Prompt) and generation step attentions.

        Args:
            all_step_attentions: The 'attentions' tuple from HuggingFace output.
        Returns:
            List of [num_heads, seq_len, seq_len] attention matrices, one per layer.
        """
        num_layers = len(all_step_attentions[0])
        num_layers = min(num_layers, self.end_layer)

        full_layer_attentions = []

        for layer_idx in range(self.start_layer, num_layers):
            prefill_attn = all_step_attentions[0][layer_idx][0]
            num_heads, prompt_len, _ = prefill_attn.shape

            total_gen_steps = len(all_step_attentions) - 1
            total_len = prompt_len + total_gen_steps

            full_matrix = torch.zeros((num_heads, total_len, total_len), device=prefill_attn.device)
            full_matrix[:, :prompt_len, :prompt_len] = prefill_attn

            for step in range(total_gen_steps):
                current_len = prompt_len + step
                step_attn = all_step_attentions[step + 1][layer_idx][0].squeeze(1)
                full_matrix[:, current_len, :current_len + 1] = step_attn

            full_layer_attentions.append(full_matrix)

        return full_layer_attentions

    def compute_rollout(self, stitched_attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the final rollout matrix with Receptive Field Normalization
        (Column-wise) to fix the causal mask geometric bias.

        Returns:
            rollout_matrix: [num_heads, seq_len, seq_len]
            rollout_matrix_all_layers: list of per-layer rollout matrices
        """
        num_heads = stitched_attentions[0].shape[0]
        seq_len = stitched_attentions[0].shape[1]
        device = stitched_attentions[0].device

        receptive_field_weights = torch.arange(1, seq_len + 1, device=device).unsqueeze(0).unsqueeze(0)
        rollout_matrix = torch.eye(seq_len, device=device).unsqueeze(0).expand(num_heads, -1, -1).clone()

        rollout_matrix_all_layers = []
        for layer_attn in stitched_attentions:
            identity = torch.eye(seq_len, device=device).unsqueeze(0)
            attn_hat = self.residual_weight * layer_attn + self.residual_weight * identity

            attn_hat_scaled = attn_hat * receptive_field_weights

            row_sums = attn_hat_scaled.sum(dim=-1, keepdim=True)
            attn_hat_norm = attn_hat_scaled / row_sums

            rollout_matrix = torch.matmul(attn_hat_norm, rollout_matrix)
            rollout_matrix_all_layers.append(rollout_matrix.clone())

        return rollout_matrix, rollout_matrix_all_layers

    def compute_rollout_simple(
        self,
        gen_dict: Dict[str, Any],
        inputs: Any,
        token_indices: Optional[Dict[str, Any]] = None,
        return_mode: str = 'o-to-img',
        do_rollout: bool = True
    ) -> np.ndarray:
        """
        Computes rollout for the generated sequence and extracts attention
        from the output tokens to the visual (image) tokens.

        Args:
            gen_dict: Generation dictionary with attentions
            inputs: Model inputs
            token_indices: Dict with img_start_idx, img_end_idx, input_len
            return_mode: Mode for extracting attention. Options:
                - 'o-to-img': Output tokens to image tokens (default)
                - 'stitched': Return raw stitched attentions
            do_rollout: Whether to apply rollout algorithm
        """
        if not gen_dict.get('attentions'):
            raise ValueError("gen_dict must contain 'attentions' (set output_attentions=True)")

        stitched_attentions = self._stitch_attention_matrices(gen_dict.attentions)

        if do_rollout:
            full_rollout_matrix, _ = self.compute_rollout(stitched_attentions)
        else:
            stitched_stack = torch.stack(stitched_attentions)
            full_rollout_matrix = stitched_stack.mean(dim=0)

        if token_indices and 'img_start_idx' in token_indices:
            img_start = token_indices['img_start_idx']
            img_end = token_indices['img_end_idx']
        else:
            IMAGE_TOKEN_IDX = 151655
            input_ids = inputs.input_ids[0].cpu().numpy()
            img_mask = input_ids == IMAGE_TOKEN_IDX
            img_indices = np.where(img_mask)[0]
            img_start, img_end = img_indices[0], img_indices[-1] + 1

        prompt_len = gen_dict.attentions[0][0].shape[-1]

        num_visual_tokens = token_indices['img_end_idx'] - token_indices['img_start_idx']
        grid_h, grid_w = self._get_grid_shape(inputs, num_visual_tokens)

        if return_mode == 'stitched':
            stitched_stack = torch.stack(stitched_attentions)
            return stitched_stack, grid_h, grid_w

        # 'o-to-img' (default): output tokens to image tokens
        q_start_idx = prompt_len - 1
        q_end_idx = full_rollout_matrix.shape[1]

        output_to_image_attn = full_rollout_matrix[:, q_start_idx:q_end_idx, img_start:img_end]

        receptive_field_weights = torch.arange(1, full_rollout_matrix.shape[-1] + 1, device=output_to_image_attn.device)[q_start_idx:q_end_idx].unsqueeze(0).unsqueeze(2)
        output_to_image_attn = output_to_image_attn * receptive_field_weights / full_rollout_matrix.shape[-1]

        num_visual_tokens = output_to_image_attn.shape[-1]
        grid_h, grid_w = self._get_grid_shape(inputs, num_visual_tokens)

        avg_visual_attn = output_to_image_attn.mean(dim=0).mean(dim=0).cpu().numpy()
        return avg_visual_attn.reshape(grid_h, grid_w)

    def _get_grid_shape(self, inputs: Any, num_visual_tokens: int) -> Tuple[int, int]:
        """Get grid shape from inputs or infer from number of tokens."""
        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            t, h, w = inputs.image_grid_thw[0].tolist()
            spatial_merge_size = 2
            grid_h = h // spatial_merge_size
            grid_w = w // spatial_merge_size
        else:
            grid_size = int(np.sqrt(num_visual_tokens))
            grid_h = grid_w = grid_size

        return grid_h, grid_w
