export CUDA_VISIBLE_DEVICES='4,5,6,7'

MODEL_NAME='meta-llama/Llama-2-7b-hf'
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'
OP_DIR="${PROJECT_DIR}/open-instruct"
ENCODED_DATASETS_PATH="${PROJECT_DIR}/datasets/sft-encoded-datasets"
OUTPUT_DIR="${PROJECT_DIR}/models"
MAX_SEQ_LENGTH='1024'
CHECKPOINTING_STEPS='100' # 'epoch'

nohup \
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file "${OP_DIR}/ds_configs/stage3_offloading_accelerate.conf" \
    "${OP_DIR}/open_instruct/finetune.py" \
    --use_flash_attn \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
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
    --output_dir "${OUTPUT_DIR}/${MODEL_NAME}" \
    --with_tracking \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --report_to wandb \
    --logging_steps 1 \
    &
