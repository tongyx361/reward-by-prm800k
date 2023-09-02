PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'

EVAL_SCRIPT_PATH=/data/users/zhangjunlei/tyx/reward-by-prm800k/src/test-eval.py
# MODEL_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
MODEL_PATH="/data/users/zhangjunlei/tyx/reward-by-prm800k/models/llama2-7b-direct-prediction/step_2000_fp16_hf"
TOKENIZER_PATH="/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470"
DATASETS_NAME="train+validiation-direct-prediction-encoded-datasets"
BATCH_SIZE_PER_GPU=4 # <8
NUM_GPUS=4
MAX_SEQ_LENGTH='1024'

ENCODED_DATASETS_PATH="${PROJECT_DIR}/datasets/${DATASETS_NAME}"

CUDA_VISIBLE_DEVICES="3,4,5,6" \
nohup accelerate launch \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    "${EVAL_SCRIPT_PATH}"\
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name "${TOKENIZER_PATH}" \
    --encoded_datasets_name_or_path "${ENCODED_DATASETS_PATH}" \
    --per_device_eval_batch_size "${BATCH_SIZE_PER_GPU}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --prm800k \
    &> "${PROJECT_DIR}/logs/test-eval.log" \
    &

    # --use_deepspeed \