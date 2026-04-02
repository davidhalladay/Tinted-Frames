"""
Model Adapter Configuration

Maps each supported model key to its architecture-specific details:
layer paths, embedding paths, image token IDs, processor quirks, etc.

This is a lightweight config dict approach — no class hierarchy needed.
"""

from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-model architecture configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'qwen2.5vl': {
        'layer_path': 'model.language_model.layers',
        'embed_path': 'model.language_model.embed_tokens',
        'attn_suffix': 'self_attn',
        'num_layers': 28,
        'image_token_id': 151655,
        'supports_output_attentions': True,
        'has_grid_thw': True,
        'grid_thw_divisor': 2,  # grid_h = thw[1]//2, grid_w = thw[2]//2
        'attn_implementation': 'eager',
        'model_class': 'Qwen2_5_VLForConditionalGeneration',
        'model_import': 'transformers',
        'vision_encoder': {
            'vit_blocks': 'visual.blocks',
            'mlp_type': 'swiglu',
            'mlp_down_proj': 'mlp.down_proj',
            'merger': 'visual.merger',
            'vit_forward': 'visual',
        },
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get the config dict for a given model key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]


def get_vision_encoder_config(model_key: str) -> Optional[Dict[str, Any]]:
    """Get the vision encoder sub-config for a given model key, or None if not defined."""
    cfg = get_model_config(model_key)
    return cfg.get('vision_encoder')


def get_lora_target_modules(config: Dict[str, Any], layer_start: int, layer_end: int) -> List[str]:
    """
    Return list of LoRA target module name strings for layers [layer_start, layer_end).

    These are full dotted paths relative to the model root, e.g.:
        'model.language_model.layers.10.self_attn.q_proj'
    """
    layer_path = config['layer_path']
    attn_suffix = config.get('attn_suffix', 'self_attn')
    target_modules = []
    for i in range(layer_start, layer_end):
        prefix = f"{layer_path}.{i}.{attn_suffix}"
        target_modules += [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
    return target_modules


def get_embed_layer(model, config: Dict[str, Any]):
    """
    Navigate the model to return the embedding nn.Module.

    e.g. for embed_path='model.language_model.embed_tokens',
    returns model.model.language_model.embed_tokens

    Note: embed_path is relative to the model root (the top-level nn.Module),
    so we just traverse using getattr.
    """
    try:
        parts = config['embed_path'].split('.')
        obj = model
        for part in parts:
            obj = getattr(obj, part)
    except AttributeError as e:
        print(f"Error navigating to embed layer with path '{config['embed_path']}': {e}")
        config['embed_path'] = config['embed_path'].replace(".language_model", "")
        parts = config['embed_path'].split('.')
        obj = model
        for part in parts:
            obj = getattr(obj, part)
    return obj


def get_image_token_id(processor, config: Dict[str, Any]) -> int:
    """
    Return the image token ID for the given model.

    If config specifies a fixed ID, use that.
    Otherwise, try to resolve from the processor/tokenizer.
    """
    fixed_id = config.get('image_token_id')
    if fixed_id is not None:
        return fixed_id

    # Fallback: try image_token_id attribute
    tokenizer = getattr(processor, 'tokenizer', processor)
    if hasattr(tokenizer, 'image_token_id'):
        return tokenizer.image_token_id
    if hasattr(processor, 'image_token_id'):
        return processor.image_token_id

    # Try common token names
    for token_name in ['<image>', '<|image|>', '<img>', '<image_pad>', '<|vision_start|>']:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id != tokenizer.unk_token_id:
            return token_id

    raise ValueError(
        f"Could not determine image_token_id for model. "
        f"Set it explicitly in MODEL_CONFIGS or check the processor/tokenizer."
    )


def get_grid_hw(processor_output, config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Extract (grid_H, grid_W) from processor output if the model uses image_grid_thw.

    Returns None for models that don't have grid_thw.
    """

    if not config.get('has_grid_thw', False):
        return None

    if not hasattr(processor_output, 'image_grid_thw') or processor_output.image_grid_thw is None:
        return None

    thw = processor_output.image_grid_thw[0]  # first image
    divisor = config.get('grid_thw_divisor', 2)
    return (int(thw[1]) // divisor, int(thw[2]) // divisor)
