PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'

nohup python "${PROJECT_DIR}/src/keep-process.py" \
    --script_path "${PROJECT_DIR}/src/download-from-hf-hub.py" \
    --process_wise_args_path "${PROJECT_DIR}/scripts/model_name.txt" \
    &> "${PROJECT_DIR}/logs/download-models.out" \
    &