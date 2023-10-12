# `config.json` should be cp to model variant dir, e.g. `best-wizardmath-13b/`
model_rel_path="best-wizardmath-13b/step_900"
PROJECT_HOME="/data/users/zhangjunlei/tyx/reward-by-prm800k"
nohup python "$PROJECT_HOME/src/eval-models-with-best-of-n-entry.py" \
    --model_name_or_path "$PROJECT_HOME/models/$model_rel_path" \
    --models_dirpath "$PROJECT_HOME/models/best-wizardmath-13b" \
    --gpu_ids "0,1" \
    > "$PROJECT_HOME/logs/eval-models-with-best-of-n.log" 2>&1 &