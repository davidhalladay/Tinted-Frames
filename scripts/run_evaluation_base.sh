#!/bin/bash
# Section 5: Benchmark evaluation of finetuned models

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/evaluation/baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
FINETUNE_MODE="${FINETUNE_MODE:-none}" # none ptuning_8_postfix

AUG_ARG=""
if [ -n "$CHECKPOINT_PATH" ]; then
    AUG_ARG="--aug_model_path $CHECKPOINT_PATH"
fi

mkdir -p "$OUTPUT_DIR"

echo "======================================== realworldqa"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset realworldqa \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/realworldqa"

echo "======================================== mme"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset mme \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/mme"

echo "======================================== mmmupro"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset mmmupro \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/mmmupro"

echo "======================================== hallusion"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset hallusion \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/hallusion"

echo "======================================== pope"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset pope \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/pope"

echo "======================================== hrbench8k"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset hrbench8k \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/hrbench8k"

echo "======================================== vstar"
python -m analysis.evaluate_finetuned \
    --model_path "$MODEL_PATH" \
    --dataset vstar \
    --finetune_mode "$FINETUNE_MODE" \
    $AUG_ARG \
    --output_dir "$OUTPUT_DIR/vstar" 
