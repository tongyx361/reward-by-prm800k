# every run
export CUDA_VISIBLE_DEVICES="0,1,3,5"
NUM_GPUS=4
# NUM_GPUS=2
DATE="2023-10-07"
IDX=3
# DEBUG_SUFFIX="-debug"

unset RESUME_TRACKING
# RESUME_TRACKING="must"

unset RUN_ID
# RUN_ID="2r9crs86"

unset RESUME_CKPT_PATH
# RESUME_CKPT_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/llama2-13b-direct-prediction-002validation-bs=128-gas=16/step_2600"
# RESUME_CKPT_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/wizardmath-13b-prm800k-train-direct-prediction-0-02validiation-bs=128-gas=16/epoch_1"

# every exp

# MODEL_NAME="llama2"
# MODEL_NAME="wizardmath"
MODEL_NAME='mammoth'

unset PEFT
# PEFT="lora"

# MODEL_SIZE_B=70
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--TheBloke--Llama-2-70B-fp16/snapshots/b25061ef1b440e970d15d4ac99bc42937cd442a2"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--WizardLM--WizardMath-70B-V1.0/snapshots/e089c3f9d2ad9d1acb62425aec3f4126f498f4c5"


MODEL_SIZE_B="13"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--WizardLM--WizardMath-13B-V1.0/snapshots/7ef412d2c680ef0fbdcd88d0df31b396d8d3049c"
MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--TIGER-Lab--MAmmoTH-13B/snapshots/722090488fd2fd201877cfb73e831bd7669b6397"
# MODEL_PATH="WizardLM/WizardMath-13B-V1.0"


# MODEL_SIZE_B="7"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"


# MODEL_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/llama2-7b-direct-prediction/step_2000_fp16_hf"

TOKENIZER_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470"

# ACCELERATE_LIB="flash-attn-v2"
ACCELERATE_LIB="xformers"
# ACCELERATE_LIB="flash-attn-v1"
# ACCELERATE_LIB=""



# ENCODED_DATASETS_NAME="encoded-datasets-direct-prediction"
DATASET_NAME="prm800k-train-direct-prediction-0-02validiation"
ENCODED_DATASETS_NAME="${DATASET_NAME}-encoded-datasets"
TRAIN_BATCH_SIZE_PER_GPU=2
# TRAIN_BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
# TOTAL_BATCH_SIZE=120
# TOTAL_BATCH_SIZE=96
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$TRAIN_BATCH_SIZE_PER_GPU))
NUM_TRAIN_EPOCHS=5

# 以能继续训练为同一个实验

HYPER_PARAMS_INFO="bs=${TOTAL_BATCH_SIZE}-gas=${GRADIENT_ACC_STEPS}"
if [ -z "${PEFT}" ]; then
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE_B}b-${DATASET_NAME}-${HYPER_PARAMS_INFO}"
else
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE_B}b-${PEFT}-${DATASET_NAME}-${HYPER_PARAMS_INFO}"
fi

export WANDB_NAME="${EXP_NAME}-${DATE}-${IDX}"

# MAX_SEQ_LENGTH='4096'
MAX_SEQ_LENGTH='1024'

# seldom change
CHECKPOINTING_STEPS=''
# CHECKPOINTING_STEPS=100
# EVAL_STEPS=100
# EVAL_STEPS=20
EVAL_STEPS=50

PROJECT_NAME="step-reward"

PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'
# OP_DIR="${PROJECT_DIR}/open-instruct"

# DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_offloading_accelerate.conf"
# DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_no_offloading.conf"
DS_CONFIG_PATH="${PROJECT_DIR}/ds_configs/stage3_no_offloading.conf"

echo "Training ${MODEL_SIZE_B}B model using $NUM_GPUS GPUs, $TRAIN_BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
ENCODED_DATASETS_PATH="${PROJECT_DIR}/datasets/${ENCODED_DATASETS_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/models/${EXP_NAME}"
LOG_DIR="${PROJECT_DIR}/logs/train/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
LOG_NAME="${DATE}-${IDX}.out"
LOG_PATH="${LOG_DIR}/${LOG_NAME}"

# command

nohup \
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    --use_deepspeed \
    --deepspeed_config_file "${DS_CONFIG_PATH}" \
    "${PROJECT_DIR}/src/finetune.py" \
    --project_name "${PROJECT_NAME}" \
    --model_name_or_path "${MODEL_PATH}" \
    --resume_from_checkpoint "${RESUME_CKPT_PATH}" \
    --use_accelerate_lib "${ACCELERATE_LIB}" \
    --tokenizer_name "${TOKENIZER_PATH}" \
    --use_slow_tokenizer \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --encoded_datasets_name_or_path "${ENCODED_DATASETS_PATH}" \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE_PER_GPU}" \
    --per_device_eval_batch_size "${TRAIN_BATCH_SIZE_PER_GPU}" \
    --gradient_accumulation_steps "${GRADIENT_ACC_STEPS}" \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --output_dir "${OUTPUT_DIR}" \
    --with_tracking \
    --resume_tracking "${RESUME_TRACKING}" \
    --resume_run_id "${RUN_ID}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --epoch_checkpointing \
    --save_best_metric "f1_-1" \
    --report_to "wandb" \
    --logging_steps 1 \
    --data_seed 42 \
    --do_eval \
    --eval_first \
    --eval_steps "${EVAL_STEPS}" \
    --prm800k \
    --peft "${PEFT}" \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    &> "${LOG_PATH}" \
    &
    # --low_cpu_mem_usage \
    # --use_flash_attn \
    # --sync_cache_flush \
    # --resume_from_checkpoint "latest" \
    # --class_average "weighted" \
    # --debug \
    # --use_cache \
    # "${OP_DIR}/open_instruct/finetune.py" \
    #
