"""
Attention Metrics — BBox Attention, Sink Token Analysis, Visual Energy

Provides analysis utilities for measuring attention properties:
- BBoxAttentionAnalyzer: attention within bounding boxes
- SinkTokenAnalyzer: attention to sink tokens
- VisualEnergyCalculator: visual energy, entropy, dispersion
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class BBoxAttentionAnalyzer:
    """Analyzes attention within bounding boxes."""

    def __init__(self):
        pass

    def compute_bbox_attention_by_grid(
        self,
        attn_map: np.ndarray,
        bbox,
        image_size: Tuple[int, int],
        normalize: bool = False,
    ) -> float:
        """Compute total attention within bounding box (grid-space mapping)."""
        total_bbox_attention = 0.0
        if isinstance(bbox, dict):
            all_bbox = [bbox]
        else:
            all_bbox = bbox
        for single_bbox in all_bbox:
            grid_h, grid_w = attn_map.shape
            attn_map_n_grid = grid_h * grid_w
            img_w, img_h = image_size
            x = single_bbox['x'] / img_w * grid_w
            y = single_bbox['y'] / img_h * grid_h
            w = single_bbox['w'] / img_w * grid_w
            h = single_bbox['h'] / img_h * grid_h
            x_start = int(max(0, min(x, grid_w - 1)))
            y_start = int(max(0, min(y, grid_h - 1)))
            x_end = max(0, min(math.ceil(x + w), grid_w))
            y_end = max(0, min(math.ceil(y + h), grid_h))
            bbox_attn = attn_map[y_start:y_end, x_start:x_end]
            if normalize:
                bbox_attn = bbox_attn / (attn_map_n_grid + 1e-8)
            total_bbox_attention += float(bbox_attn.sum())
        return total_bbox_attention

class VisualEnergyCalculator:
    """Computes visual energy from attention maps."""

    def __init__(self, method: str = 'standard'):
        self.method = method

    def compute_visual_energy_simple(self, attention_weights: np.ndarray, normalize: bool = True) -> float:
        """Simplified interface — compute visual energy from a flat or 2D attention map."""
        attention_weights = attention_weights.flatten()
        if normalize:
            return float(np.sum(attention_weights) / len(attention_weights))
        return float(np.sum(attention_weights))

    def compute_entropy(self, attention_weights: np.ndarray) -> float:
        """Shannon entropy of the attention map (normalized to sum=1)."""
        attn = attention_weights.flatten().astype(np.float64)
        attn = attn / (attn.sum() + 1e-12)
        mask = attn > 0
        return float(-np.sum(attn[mask] * np.log(attn[mask])))

    def compute_dispersion(self, attention_weights: np.ndarray) -> float:
        """Spatial dispersion: weighted standard deviation of 2D grid positions."""
        H, W = attention_weights.shape
        attn = attention_weights.astype(np.float64)
        attn = attn / (attn.sum() + 1e-12)
        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        row_com = float(np.sum(attn * rows))
        col_com = float(np.sum(attn * cols))
        dist2 = (rows - row_com) ** 2 + (cols - col_com) ** 2
        return float(np.sqrt(np.sum(attn * dist2)))
