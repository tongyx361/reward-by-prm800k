# every run
export CUDA_VISIBLE_DEVICES='3,4,5,6'
DATE="2023-08-28"
IDX=2

# every exp
MODEL_SIZE_B="13"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
# RESUME_CKPT_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/llama2-13b-direct-prediction/step_300"
ACCELERATE_LIB="flash-attn-v2"
# ACCELERATE_LIB="flash-attn-v1"
# ACCELERATE_LIB=""

DATASETS_NAME="encoded-datasets-direct-prediction"
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
# BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
# TOTAL_BATCH_SIZE=120
# TOTAL_BATCH_SIZE=96
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

EXP_NAME="llama2-${MODEL_SIZE_B}b-direct-prediction-bs=${TOTAL_BATCH_SIZE}-gas=${GRADIENT_ACC_STEPS}" # 以能继续训练为同一个实验

# MAX_SEQ_LENGTH='4096'
MAX_SEQ_LENGTH='1024'

# may change
# DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_offloading_accelerate.conf"
DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_no_offloading.conf"

CHECKPOINTING_STEPS='100' # 'epoch'

export WANDB_PROJECT="step-reward"
export WANDB_NAME="${DATE}-${IDX}"

PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'
OP_DIR="${PROJECT_DIR}/open-instruct"

echo "Training ${MODEL_SIZE_B}B model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
ENCODED_DATASETS_PATH="${PROJECT_DIR}/datasets/${DATASETS_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/models/${EXP_NAME}"
LOG_DIR="${PROJECT_DIR}/logs/train/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
LOG_NAME="${DATE}-${IDX}.out"
LOG_PATH="${LOG_DIR}/${LOG_NAME}"


nohup \
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file "${DS_CONFIG_PATH}" \
    "${OP_DIR}/open_instruct/finetune.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --use_accelerate_lib "${ACCELERATE_LIB}" \
    --tokenizer_name "${MODEL_PATH}" \
    --use_slow_tokenizer \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --encoded_datasets_name_or_path "${ENCODED_DATASETS_PATH}" \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir "${OUTPUT_DIR}" \
    --with_tracking \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --report_to "wandb" \
    --logging_steps 1 \
    --resume_from_checkpoint "" \
    &> "${LOG_PATH}" \
    &

    # --use_flash_attn \
    # --sync_cache_flush \
    # --resume_from_checkpoint "latest" \