PROJECT_DIR='/data/users/zhangjunlei/tyx/reward-by-prm800k'

nohup python "${PROJECT_DIR}/src/keep-process.py" \
    --script_path "${PROJECT_DIR}/src/download-from-hf-hub.py" \
    --process_wise_args_path "${PROJECT_DIR}/scripts/model_name.txt" \
    --max_trials 100 \
    --trial_interval 60 \
    &> "${PROJECT_DIR}/logs/download-models.out" \
    &