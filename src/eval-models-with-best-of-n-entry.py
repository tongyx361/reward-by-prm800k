'''iterate with eval-one-model-with-best-of-n.py'''
# %%
import argparse
import os
import subprocess

import regex as re
import torch
import utils


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=True)

    # 添加可能出现的命令行参数
    parser.add_argument('--model_name_or_path', type=str, default=None, help="Path to the model to evaluate.")
    parser.add_argument('--models_dirpath', type=str, default=None, help="Metrics to evaluate.")
    parser.add_argument('--gpu_ids', type=str, default="0", help="IDs of GPUs to use.")

    # 解析参数
    args = parser.parse_args()
    return args


args = parse_args()

# gpu_ids = [1, 3]
gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
num_gpus = len(gpu_ids)
utils.set_gpu_ids(gpu_ids)

# models_dirpath = "/data/users/zhangjunlei/tyx/reward-by-prm800k/models/direct-prediction/meta-llama/Llama-2-7b-hf"
# models_dirpath = "/data/users/zhangjunlei/tyx/reward-by-prm800k/models/best-wizardmath-13b/step=900-best-f1_-1=0.5783492822966507"

# output: utils.best_of_n_results_jsonl_path

# %%
if args.model_name_or_path is not None:
    model_variant_paths = [args.model_name_or_path]
elif args.models_dirpath is not None:
    models_dirpath = args.models_dirpath
    model_variant_paths = []
    for name in os.listdir(models_dirpath):
        path = os.path.join(models_dirpath, name)
        if not os.path.isdir(path):
            continue
        if not utils.is_zero3_parameters(path):
            continue
        model_variant_paths.append(path)
else:
    raise ValueError("Either --model_name_or_path or --models_dirpath must be specified.")


def extract_step_or_epoch_num(path):
    return int(re.search(r".*?(?:step|epoch)[_=-]?([0-9]+)?", path).group(1))


if args.model_name_or_path is not None:
    model_variant_paths.sort(key=extract_step_or_epoch_num, reverse=True)

# model_variant_paths = [
#     model_variant_path
#     for model_variant_path in model_variant_paths
#     # if 1 < extract_step_or_epoch_num(model_variant_path) < 100
#     # or extract_step_or_epoch_num(model_variant_path) >= 100
# ]

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
            args.gpu_ids,
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
