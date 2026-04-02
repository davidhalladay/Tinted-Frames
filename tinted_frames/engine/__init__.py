"""
Engine module for VLM inference and attention analysis.
"""

from .attention import AttentionRollout
from .inference import InferenceEngine, ModelManager
from .metrics import BBoxAttentionAnalyzer, VisualEnergyCalculator
from .model_config import (
    MODEL_CONFIGS, get_model_config, get_lora_target_modules,
    get_embed_layer, get_image_token_id, get_grid_hw,
    get_vision_encoder_config,
)

__all__ = [
    'AttentionRollout',
    'InferenceEngine',
    'ModelManager',
    'MODEL_CONFIGS',
    'get_model_config',
    'get_lora_target_modules',
    'get_embed_layer',
    'get_image_token_id',
    'get_grid_hw',
    'get_vision_encoder_config',
    'BBoxAttentionAnalyzer',
    'VisualEnergyCalculator',
]
