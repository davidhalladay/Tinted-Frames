#!/bin/bash
# Run evaluations for baseline and all trained model checkpoints

echo "########################################## Baseline
FINETUNE_MODE="none" \
CHECKPOINT_PATH="" \
OUTPUT_DIR="checkpoints/evaluation/baseline" \
bash scripts/run_evaluation_base.sh

echo "########################################## Ours
FINETUNE_MODE="ptuning_8_postfix" \
CHECKPOINT_PATH="checkpoints/qwen25vl7b/" \
OUTPUT_DIR="checkpoints/evaluation/ours" \
bash scripts/run_evaluation_base.sh

python3 tinted_frames/utils/results_to_tsv_batch.py ./checkpoints/evaluation/