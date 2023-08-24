# %%
import importlib

import utils

importlib.reload(utils)

gpu_ids = [4, 5]
num_gpus = len(gpu_ids)
utils.set_gpu_ids(gpu_ids)

import multiprocessing as mp

# %%
import os
import subprocess

import regex as re
import torch

# %%
# input
train_models_dirpath = "/data/users/zhangjunlei/tyx/reward-by-prm800k/models/direct-prediction/meta-llama/Llama-2-7b-hf"

# output: utils.best_of_n_results_jsonl_path

# %%
model_variant_paths = []
for name in os.listdir(train_models_dirpath):
    path = os.path.join(train_models_dirpath, name)
    if not os.path.isdir(path):
        continue
    if not utils.is_zero3_parameters(path):
        continue
    model_variant_paths.append(path)


def extract_step_or_epoch_num(path):
    return int(re.search(r".*?(?:step|epoch)_([0-9]+)", path).group(1))


model_variant_paths.sort(key=extract_step_or_epoch_num, reverse=True)

model_variant_paths = [
    model_variant_path
    for model_variant_path in model_variant_paths
    # if 1 < extract_step_or_epoch_num(model_variant_path) < 100
    # or extract_step_or_epoch_num(model_variant_path) >= 100
]

print(model_variant_paths)

# %%
for model_variant_path in model_variant_paths:
    eval_process_results = subprocess.run(
        args=[
            utils.python_path,
            "/data/users/zhangjunlei/tyx/reward-by-prm800k/src/eval-one-model-with-best-of-n.py",
            "--model_name_or_path",
            model_variant_path,
            "--gpu_ids",
            ",".join([str(gpu_id) for gpu_id in gpu_ids]),
        ]
    )
    if eval_process_results.returncode != 0:
        raise RuntimeError(f"Failed to evaluate model variant {model_variant_path}.")

# %%
# def process_fn(gpu_id, func, *args, **kwargs):
#     torch.cuda.set_device(gpu_id)
#     func(*args, **kwargs)


# # if __name__ == '__main__':
# with mp.Pool(num_gpus) as pool:
#     pool.map(process_fn, gpu_ids)

# %%
# model_variant_path = "/data/users/zhangjunlei/tyx/reward-by-prm800k/models/direct-prediction/meta-llama/Llama-2-7b-hf/step_2000"

# utils.eval_model_with_best_of_n(
#     model_name_or_path=model_variant_path,
#     metrics=[metric for metric in utils.all_metrics if metric != "majority_voting"],
#     debug_for={"resume_vllm_outputs": True},
# )

# %%
