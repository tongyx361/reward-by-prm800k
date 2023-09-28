# every run
unset TRANSFORMERS_OFFLINE
# export TRANSFORMERS_OFFLINE=1

# wandb

unset WANDB_TAGS
WANBD_TAGS="important"
unset WANDB_MODE
# export WANDB_MODE='offline'

# devices

# DEVICE_IDS="0,4,5,6"
DEVICE_IDS="0,1"
unset CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES="0,1,3,4"
# NUM_GPUS=4
# NUM_GPUS=2
NUM_GPUS=$(echo "$DEVICE_IDS" | tr ',' '\n' | grep -c '^[0-9]*$')
# DATE="2023-09-24"
# IDX=10
TIME_STAMP=$(date +"%Y-%m-%d-%H-%M-%S")
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# DEBUG='True'
DEBUG='False'

EVAL_FIRST='True'
# EVAL_FIRST='False'
# DS_CONFIG_NAME="stage0.conf"
# DS_CONFIG_NAME="stage1.conf"
# DS_CONFIG_NAME="stage2.conf"
DS_CONFIG_NAME="stage3_no_offloading.conf"

if [ ${DEBUG} = 'True' ]; then
    export WANBD_TAGS="debug,${WANBD_TAGS}"
fi

unset WANDB_RESUME
# export WANDB_RESUME="must"

unset WANDB_RUN_ID
# export WANDB_RUN_ID="h9qgev7w"

unset RESUME_CKPT_PATH
# RESUME_CKPT_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/llama2-13b-direct-prediction-002validation-bs=128-gas=16/step_2600"
# RESUME_CKPT_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/wizardmath-13b-prm800k-train-direct-prediction-0-02validiation-bs=128-gas=16/epoch_5"

# every exp

# MODEL_NAME="llama2"
# MODEL_NAME="wizardmath"
MODEL_NAME="mammoth"

# MODEL_SIZE_B=70
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--TheBloke--Llama-2-70B-fp16/snapshots/b25061ef1b440e970d15d4ac99bc42937cd442a2"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--WizardLM--WizardMath-70B-V1.0/snapshots/e089c3f9d2ad9d1acb62425aec3f4126f498f4c5"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--TIGER-Lab--MAmmoTH-70B/snapshots/350ab0bc95339af5e17d38f3176811b77b7d7a16"

# MODEL_SIZE_B="13"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--WizardLM--WizardMath-13B-V1.0/snapshots/7ef412d2c680ef0fbdcd88d0df31b396d8d3049c"
# MODEL_PATH="WizardLM/WizardMath-13B-V1.0"

MODEL_SIZE_B="7"
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--TIGER-Lab--MAmmoTH-7B/snapshots/a177fafdaf348e229323e711adc8da388c5993b6"

TOKENIZER_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470"

unset PEFT
# PEFT="lora"
# PEFT="qlora"
if [[ ${PEFT} == 'qlora' ]]; then
    export WANDB_TAGS="qlora,${WANBD_TAGS}"
    QLORA="True"
else
    QLORA="False"
fi

# ACCELERATE_LIB="flash-attn-v2"
ACCELERATE_LIB="xformers"
# ACCELERATE_LIB="flash-attn-v1"
# ACCELERATE_LIB=""

# ENCODED_DATASETS_NAME="encoded-datasets-direct-prediction"
DATASETS_NAME="prm800k-train-direct-prediction-0-02validiation"
ENCODED_DATASETS_NAME="${DATASETS_NAME}-encoded-datasets"
# TRAIN_BATCH_SIZE_PER_GPU=16
TRAIN_BATCH_SIZE_PER_GPU=2 # 7B ZeRO-3 >= 2
# EVAL_BATCH_SIZE_PER_GPU=4
EVAL_BATCH_SIZE_PER_GPU=8 # 7B >= 128, but 8/16/128 have almost the same performance
TOTAL_BATCH_SIZE=128
# TOTAL_BATCH_SIZE=120
# TOTAL_BATCH_SIZE=96
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $TRAIN_BATCH_SIZE_PER_GPU))
NUM_TRAIN_EPOCHS=100

# LR="2e-5"
LR="1e-4"
# LR_SCHEDULE="linear"
LR_SCHEDULE="constant"
WEIGHT_DECAY=0
WARMUP=0.03

LORA_MODULES="all_linear"
LORA_R=16
LORA_ALPHA=64
LORA_DROPOUT=0.05

# 以能继续训练为同一个实验

# experiment name

LR_INFO="lr=${LR}-schdule=${LR_SCHEDULE}-wd=${WEIGHT_DECAY}-warmup=${WARMUP}"
LORA_INFO="lora-modules=${LORA_MODULES}-r=${LORA_R}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}"
HYPER_PARAMS_INFO="bs=${TOTAL_BATCH_SIZE}-gas=${GRADIENT_ACC_STEPS}-${LR_INFO}"
if [ -z "${PEFT}" ]; then
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE_B}b-${DATASETS_NAME}-${HYPER_PARAMS_INFO}"
else
    if [[ "${PEFT}" =~ "lora" ]]; then
        HYPER_PARAMS_INFO="${HYPER_PARAMS_INFO}-${LORA_INFO}"
    fi
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE_B}b-${PEFT}-${DATASETS_NAME}-${HYPER_PARAMS_INFO}"
fi

# export WANDB_NAME="${DATE}-${IDX}-${EXP_NAME}"
export WANDB_NAME="${TIME_STAMP}-${EXP_NAME}"

# MAX_SEQ_LENGTH=4096
MAX_SEQ_LENGTH=1024

# seldom change
unset CHECKPOINTING_STEPS
# CHECKPOINTING_STEPS=100
EVAL_STEPS=100

PROJECT_NAME="step-reward"
export WANDB_PROJECT="${PROJECT_NAME}"

PROJECT_DIR="${DATA_ROOT}/reward-by-prm800k"

# OP_DIR="${PROJECT_DIR}/open-instruct"
# DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_offloading_accelerate.conf"
# DS_CONFIG_PATH="${OP_DIR}/ds_configs/stage3_no_offloading.conf"

DS_CONFIG_DIR="${PROJECT_DIR}/ds_configs"
DS_CONFIG_PATH="${DS_CONFIG_DIR}/${DS_CONFIG_NAME}"

echo "Training ${MODEL_SIZE_B}B model using $NUM_GPUS GPUs, $TRAIN_BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
ENCODED_DATASETS_PATH="${PROJECT_DIR}/datasets/${ENCODED_DATASETS_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/models/${EXP_NAME}"
LOG_DIR="${PROJECT_DIR}/logs/train/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
# LOG_NAME="${DATE}-${IDX}.out"
# LOG_PATH="${LOG_DIR}/${LOG_NAME}"
LOG_PATH="${LOG_DIR}/${TIME_STAMP}.out}"

if [[ ${ACCELERATE_LIB} = "flash-attn-v2" ]]; then
    TRAIN_SCRIPT_PATH="${PROJECT_DIR}/src/train_flash_attn_2.py"
elif [[ ${ACCELERATE_LIB} = "xformers" ]]; then
    TRAIN_SCRIPT_PATH="${PROJECT_DIR}/src/train_xformers.py"
else
    TRAIN_SCRIPT_PATH="${PROJECT_DIR}/src/train.py"
fi

# command

# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes "${NUM_GPUS}" \
#     --use_deepspeed \
#     --deepspeed_config_file "${DS_CONFIG_PATH}" \
nohup \
    deepspeed \
    --include "localhost:${DEVICE_IDS}" \
    "${TRAIN_SCRIPT_PATH}" \
    --model_name_or_path "$MODEL_PATH" \
    --encoded_datasets_path "$ENCODED_DATASETS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_max_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE_PER_GPU" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE_PER_GPU" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps "$GRADIENT_ACC_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --eval_first ${EVAL_FIRST} \
    --evaluation_strategy "steps" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy "steps" \
    --save_steps "$EVAL_STEPS" \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "f1_-1" \
    --greater_is_better True \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP}" \
    --lr_scheduler_type "${LR_SCHEDULE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_MODULES}" \
    --q_lora ${QLORA} \
    --bf16 True \
    --tf32 True \
    --use_accelerate_lib "${ACCELERATE_LIB}" \
    --deepspeed "${DS_CONFIG_PATH}" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --custom_debug "${DEBUG}" \
    &>"$LOG_PATH" \
    &
# --fp16 True \
# --save_total_limit 2 \
# --flash_attn False
# --lora_r 256 \
# --lora_alpha 256 \
# --eval_first False \
