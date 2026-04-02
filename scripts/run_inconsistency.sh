#!/bin/bash
# Section 3: Cross-framing inconsistency analysis
# Analyzes how different question framings lead to inconsistent answers

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inconsistency}"
N_SAMPLES="${N_SAMPLES:-3000}" # Set to 3000 for testing
SEED="${SEED:-42}"
DATASET="${1:-gqa}"

GQA_QUESTIONS="data/gqa/data/val_pos_multiobj_balanced_questions_full.json"
GQA_SCENE_GRAPHS="data/gqa/val_sceneGraphs.json"
GQA_IMAGES="data/gqa/images"
SEEDBENCH_ANSWERS="data/seed_bench/SEED-Bench.json"
SEEDBENCH_IMAGES="data/seed_bench/SEED-Bench-image"

OUTPUT_DIR="$OUTPUT_DIR/${DATASET}_samples${N_SAMPLES}"
mkdir -p "$OUTPUT_DIR"

if [ "$DATASET" = "gqa" ]; then
    python -m analysis.cross_framing_inconsistency \
        --model_path "$MODEL_PATH" \
        --dataset gqa \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --gqa-questions-json "$GQA_QUESTIONS" \
        --gqa-scene-graphs "$GQA_SCENE_GRAPHS" \
        --gqa-images "$GQA_IMAGES"
else
    python -m analysis.cross_framing_inconsistency \
        --model_path "$MODEL_PATH" \
        --dataset seedbench \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        --seedbench-answers-json "$SEEDBENCH_ANSWERS" \
        --seedbench-images "$SEEDBENCH_IMAGES"
fi
