PROCESS_NAME="openai-api"

nohup python "${WORK_REPO_PATH}/src/keep-process.py" \
    --script_path "${WORK_REPO_PATH}/src/${PROCESS_NAME}.py" \
    --trial_interval 30 \
    &> "${WORK_REPO_PATH}/logs/${PROCESS_NAME}.out" \
    &