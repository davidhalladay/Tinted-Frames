#!/bin/bash
# Section 4: Framing attention analysis
# Analyzes how framing modulates visual attention patterns

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/attention_analysis}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"

DATA_BASE="./data"
GQA_QUESTIONS_JSON="${GQA_QUESTIONS_JSON:-${DATA_BASE}/gqa/data/val_pos_multiobj_balanced_questions_full.json}"
REFRAMED_GQA_QUESTIONS_JSON="${REFRAMED_GQA_QUESTIONS_JSON:-${DATA_BASE}/gqa_framing/reframed_questions.json}"
GQA_IMAGES="${GQA_IMAGES:-${DATA_BASE}/gqa/images}"
GQA_SCENE_GRAPHS="${GQA_SCENE_GRAPHS:-${DATA_BASE}/gqa/val_sceneGraphs.json}"
VSTAR_FRAMING_JSON="${VSTAR_FRAMING_JSON:-${DATA_BASE}/vstar_bench_framing/vstar_framing.json}"
VSTAR_IMAGE_ROOT="${VSTAR_IMAGE_ROOT:-${DATA_BASE}/vstar_bench}"

mkdir -p "$OUTPUT_DIR"

python -m analysis.framing_attention_analysis \
    --model_path "$MODEL_PATH" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --gqa_questions_json "$GQA_QUESTIONS_JSON" \
    --reframed_gqa_questions_json "$REFRAMED_GQA_QUESTIONS_JSON" \
    --gqa_images "$GQA_IMAGES" \
    --gqa_scene_graphs "$GQA_SCENE_GRAPHS" \
    --vstar_framing_json "$VSTAR_FRAMING_JSON" \
    --vstar_image_root "$VSTAR_IMAGE_ROOT"
