#!/bin/bash
# Section 5: Prompt tuning training
# Trains learnable tokens to mitigate framing-induced attention bias

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-./data/llava_train/text_files/llava_v1_5_mix665k.json}"
IMAGE_ROOT="${IMAGE_ROOT:-./data/llava_train}"
QUESTION_MAPPER="${QUESTION_MAPPER:-./data/question_mapper.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen25vl7b}"
NUM_GPUS="${NUM_GPUS:-1}"

# Training hyperparameters
TRAINING_MODE="${TRAINING_MODE:-prompt_tuning}"
NUM_SOFT_TOKENS=8
PROMPT_POSITION="infix"
SAMPLE_N=6400
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
NUM_EPOCHS=1
LR=2e-4
LAMBDA_MASS=5.0
LAMBDA_SHAPE=0.1
LAYER_START=18
LAYER_END=24
MAX_SEQ_LEN=1024
IMAGE_SIZE=728

# LoRA (only used if TRAINING_MODE=lora)
LORA_R=16
LORA_ALPHA=32

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS="
  --model_path ${MODEL_PATH}
  --data_path ${DATA_PATH}
  --image_root ${IMAGE_ROOT}
  --question_mapper_path ${QUESTION_MAPPER}
  --sample_n ${SAMPLE_N}
  --training_mode ${TRAINING_MODE}
  --lambda_mass ${LAMBDA_MASS}
  --lambda_shape ${LAMBDA_SHAPE}
  --layer_range ${LAYER_START} ${LAYER_END}
  --per_device_train_batch_size ${BATCH_SIZE}
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}
  --num_train_epochs ${NUM_EPOCHS}
  --learning_rate ${LR}
  --max_seq_len ${MAX_SEQ_LEN}
  --image_size ${IMAGE_SIZE}
  --output_dir ${OUTPUT_DIR}
  --logging_steps 1
  --save_steps 20
  --save_total_limit -1
  --viz_log_steps 20
  --viz_n_samples 10
  --no_mcq_postfix
  --report_to tensorboard
  --bf16
  --gradient_checkpointing"

if [ "$TRAINING_MODE" = "lora" ]; then
    MODE_ARGS="--lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_target_layers ${LAYER_START} ${LAYER_END}"
else
    MODE_ARGS="--num_soft_tokens ${NUM_SOFT_TOKENS} --prompt_position ${PROMPT_POSITION}"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
        -m tinted_frames.training.train_framing ${COMMON_ARGS} ${MODE_ARGS}
else
    python -m tinted_frames.training.train_framing ${COMMON_ARGS} ${MODE_ARGS}
fi
