import csv
import datetime
import gzip
import importlib
import json
import logging
import os
import pickle
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import Any, Dict, List, Optional

import blobfile as bf
import evaluate
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import orjson
import prepare_dataset
import regex as re
import torch
import transformers
import vllm

# global variables

MULTICLASS_AVERAGINGS = ["micro", "macro", "weighted"]

CLF_METRIC_NAMES = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    # "roc_auc"
]

MUL_CLF_METRIC_NAMES = [
    "f1",
    "precision",
    "recall",
]

default_max_seq_len = 1024

# logging

default_log_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
default_log_datefmt = "%Y-%m-%d %H:%M:%S"


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=default_log_fmt,
        datefmt=default_log_datefmt,
        handlers=[
            logging.StreamHandler(),
        ],
    )


def get_logger(
    name=__name__,
    fmt=default_log_fmt,
    datefmt=default_log_datefmt,
    level=logging.INFO,
    log_file_path=None,
):
    logger = logging.getLogger(name)

    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt)

    if log_file_path is not None:
        file = logging.FileHandler(args.log_path, encoding="utf-8")
        file.setLevel(level)
        file.setFormatter(formatter)
        logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)


# environmetn variables

data_root = "/data/users/zhangjunlei/tyx"
hf_home = os.path.join(data_root, ".cache/huggingface")

os.environ.update(
    {
        "DATA_ROOT": data_root,
        "HF_HOME": hf_home,
        "TRANSFORMERS_CACHE": os.path.join(hf_home, "transformers"),
        "HF_DATASETS_CACHE": os.path.join(hf_home, "datasets"),
        "HF_MODULES_CACHE": os.path.join(hf_home, "modules"),
        "HF_METRICS_CACHE": os.path.join(hf_home, "metrics"),
    }
)

project_root = os.path.join(data_root, "reward-by-prm800k")
models_root = os.path.join(project_root, "models")

default_model_name = "meta-llama/Llama-2-7b-hf"
default_model_path = os.path.join(
    hf_home,
    "hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9",
)

tokenizer_name_or_path = "/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470"
gpt4_generated_problem_solution_hierarchical_samples_path_wo_basename = os.path.join(
    project_root, "datasets/problem-solution-hierarchical-samples"
)
rated_gpt4_generated_problem_solution_hierarchical_samples_dirpath = os.path.join(
    project_root, "eval/predictions/gpt-4-generatations"
)

prm800k_validation_predictions_dirpath = os.path.join(
    project_root, "eval/predictions/prm800k-validation"
)

best_of_n_results_jsonl_path = os.path.join(
    project_root, "eval/best-of-n-results.jsonl"
)
num_trials = 10

prm800k_phase_train_jsonl_paths = [
    "prm800k-main/prm800k/data/phase1_train.jsonl",
    "prm800k-main/prm800k/data/phase2_train.jsonl",
]
prm800k_phase_train_jsonl_paths = [
    os.path.join(project_root, path) for path in prm800k_phase_train_jsonl_paths
]

raw_datasets_path = os.path.join(
    project_root,
    "datasets/train+validiation-direct-prediction-raw-datasets",
)

encoded_datasets_path = os.path.join(
    project_root,
    "datasets/train+validiation-direct-prediction-encoded-datasets",
)


## python environment

conda_env_path = "/data/users/zhangjunlei/anaconda3/envs/open-instruct"
python_path = os.path.join(conda_env_path, "bin/python")

## PRM800K

num_total_solution_samples_per_problem = 1860
num_solution_samples_to_rate_per_problem = 16
rating2word = {1: "positive", -1: "negative", 0: "neutral"}

all_metrics = [
    "majority_voting",
    "positive_probs_product",
    "positive_probs_minimum",
    "non_negative_probs_product",
    "non_negative_probs_minimum",
]

## singelton

llm = None
tokenizer = None

## model

model_max_length = 4096
top_k = 5
generation_config = dict(
    temperature=1,
    top_p=1,
    top_k=top_k,
    max_tokens=1,
    logprobs=top_k,
)

# class

Sample = Dict[str, Any]


@dataclass
class Problem:
    problem: str
    subject: str
    level: int
    unique_id: str
    ground_truth_answer: str


# {
#     "problem": "A $90^\\circ$ rotation around the origin in the counter-clockwise direction is applied to $7 + 2i.$  What is the resulting complex number?",
#     "subject": "Precalculus",
#     "level": 2,
#     "unique_id": "test/precalculus/779.json",
#     "ground_truth_answer": "-2+7i"
# }

# data collate

def get_data_collator(tokenizer, model=None, padding="longest", max_length=default_max_seq_len):
    return prepare_dataset.DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model=model,
        padding=padding,
        max_length=max_length,
    )

# device

def nvidia_smi():
    ns_result = subprocess.run(["nvidia-smi"], capture_output=True)
    output = ns_result.stdout.decode("utf-8")
    return output


def reload():
    importlib.reload(vllm)


def set_gpu_ids(gpu_ids=[4, 5, 6, 7]):
    # gpu_ids = [7]
    # gpu_ids = [4, 5, 6, 7]
    if isinstance(gpu_ids, str):
        gpu_ids = gpu_ids.split(",")

    global tensor_parallel_size
    tensor_parallel_size = len(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])

    print(f'os.environ["CUDA_VISIBLE_DEVICES"] = {os.environ["CUDA_VISIBLE_DEVICES"]}')


# json


def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)  # fallback


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def load_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


# def dump_save(module, data, path, **kwargs):


def load_json(path, data_type=dict):
    if os.path.exists(path):
        f = open(path)
        data = json.load(f)
        f.close()
        if data == None:
            data = data_type()
        return data
    else:
        return None


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w")
    json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


# pickle


def load_pickle(path, data_type=dict):
    if os.path.exists(path):
        f = open(path, "rb")
        data = pickle.load(f)
        f.close()
        if data == None:
            data = data_type()
        return data
    else:
        return None


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()


# csv
def save_csv(data, path):
    if len(data) > 1:
        last_keys = data[0].keys()
        for row in data[1:]:
            assert last_keys == row.keys()
            last_keys = row.keys()
    with open(path, "w") as f:
        writer = csv.writer(f)
        # 写入列名
        writer.writerow(data[0].keys())
        # 写入其他行
        for row in data:
            writer.writerow(row.values())


# metrics


def get_mul_clf_metrics(
    clf_metric_names=CLF_METRIC_NAMES, averages=MULTICLASS_AVERAGINGS
):
    # CLF_METRIC_NAMES = [
    #     "accuracy",
    #     "f1",
    #     "precision",
    #     "recall",
    #     # "roc_auc"
    # ]
    # MULTICLASS_AVERAGINGS = ["micro", "macro", "weighted", "none"]

    combinations = list(product(clf_metric_names, MULTICLASS_AVERAGINGS))

    mul_clf_metrics = evaluate.combine(
        [evaluate.load(name, average=average) for name, average in combinations]
    )

    return mul_clf_metrics


# visualize


def visualize_dict_value_nums_distribuition(data_dict):
    # Get the lengths of values in the dictionary
    value_lengths = [len(value) for value in data_dict.values()]

    # Create a histogram to visualize the distribution
    plt.hist(value_lengths, bins=200, edgecolor="k")

    # Add labels and title
    plt.xlabel("Length of Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Length of Dictionary Values")

    # Show the plot
    plt.show()


def test_visualize_dict_value_lengths_distribuition():
    data_dict = {"key1": [1, 2, 3, 4], "key2": [1, 2, 3, 4, 5, 6], "key3": [1, 2]}

    visualize_dict_value_nums_distribuition(data_dict)


def visualize_dict_value_lengths(data_dict):
    # Extract the lengths of values in the dictionary
    value_lengths = [len(value) for value in data_dict.values()]
    value_lengths.sort(reverse=True)

    # Extract the keys (corresponding to dictionary items)
    # short_keys = [key[: min(10, len(value_lengths))] for key in data_dict.keys()]
    keys = list(range(len(data_dict)))

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Optional: adjust figure size
    plt.bar(keys, value_lengths, color="skyblue")

    # Add labels and title
    plt.xlabel("Dictionary Keys")
    plt.ylabel("Length of List Values")
    plt.title("Lengths of List Values in Dictionary")

    # Rotate x-axis labels for better readability (optional)
    # plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.tight_layout()  # Optional: adjust layout for better display
    plt.show()


def test_visualize_dict_value_lengths():
    data_dict = {"key1": [1, 2, 3, 4], "key2": [1, 2, 3, 4, 5, 6], "key3": [1, 2]}

    visualize_dict_value_lengths(data_dict)


# PRM800K data processing


def pick_prm800k_samples(x):
    return (not x["is_quality_control_question"]) and (
        x["label"]["finish_reason"] in ["found_error", "solution"]
    )


def reformat_prm800k_sample(sample: dict) -> dict:
    problem = sample["question"]["problem"]
    step_ratings = []

    if sample["is_quality_control_question"]:
        raise RuntimeError("is_quality_control_question is True")

    label = sample["label"]
    finish_reason = label["finish_reason"]
    if finish_reason not in ["found_error", "solution"]:
        raise RuntimeError(f"finish_reason is {finish_reason}")

    steps = label["steps"]
    for step in steps:
        chosen_completion = step["chosen_completion"]
        if step["human_completion"] is not None:
            completion = step["human_completion"]
            rating = 1
        else:
            completions = step["completions"]
            if chosen_completion is not None:
                completion = completions[chosen_completion]
            else:
                for completion in completions:
                    if completion["rating"] == -1:
                        break
            rating = completion["rating"]

        step_text = completion["text"]

        if completion["flagged"] not in [None, False]:
            print(f"{sample['timestamp']} flagged: ", completion["flagged"])
            print(sample)
        step_ratings.append({"step": step_text, "rating": rating})

    reformatted_sample = {"problem": problem, "step_ratings": step_ratings}

    if finish_reason == "found_error":
        last_rating = reformatted_sample["step_ratings"][-1]["rating"]
        assert last_rating == -1, f"last step should be -1 but {last_rating}"

    return reformatted_sample


def encode_with_problem_step_ratings_format(
    reformatted_sample, tokenizer, split="train", test=False
):
    """
    Here we assume each sample has a 'step_ratings' field. Each step_rating is a dict.
    """

    step_ratings = reformatted_sample["step_ratings"]
    if len(step_ratings) == 0:
        raise ValueError("step_ratings field is empty.")

    rating2word = {1: "positive", -1: "negative", 0: "neutral"}
    rating2token_id = {
        rating: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0]
        for rating, word in rating2word.items()
    }

    problem = reformatted_sample["problem"].strip()
    problem_step_ratings_text = problem + "\n"
    sample_input_ids = tokenizer(
        problem + "\n",
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"]
    ignore_index = -100

    if split == "train":
        sample_labels = torch.ones_like(sample_input_ids) * ignore_index
    elif split == "validation":
        sample_labels = []
    else:
        raise ValueError(f"split should be train or validation but {split}")

    for step_rating in step_ratings:
        step = step_rating["step"].strip()

        problem_step_ratings_text += step + "\n"

        step_input_ids = tokenizer(
            "\n" + step + "\n",
            return_tensors="pt",
            padding=False,
            truncation=False,
            # add_special_tokens=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        step_input_ids = step_input_ids[:, 2:]  # remove "\n"
        sample_input_ids = torch.cat((sample_input_ids, step_input_ids), dim=1)

        step_rating_token_id = rating2token_id[step_rating["rating"]]
        if split == "train":
            step_labels = torch.ones_like(step_input_ids) * ignore_index
            step_labels[
                :, -1
            ] = step_rating_token_id  # set the label for the last token_id before "\n"
            sample_labels = torch.cat((sample_labels, step_labels), dim=1)
            # keep the last token for hugging face to align input_ids and labels
        elif split == "validation":
            sample_labels.append(step_rating_token_id)
        else:
            raise ValueError(f"split should be train or validation but {split}")

    if split == "validation":
        sample_labels = torch.tensor(sample_labels)

    if test:
        sample_input_ids_from_simple_call = tokenizer(
            problem_step_ratings_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        assert torch.equal(sample_input_ids, sample_input_ids_from_simple_call), (
            sample_input_ids != sample_input_ids_from_simple_call
        )

    # attention_mask = torch.ones_like(sample_input_ids)
    encoded_sample = {
        "input_ids": sample_input_ids.flatten(),
        "labels": sample_labels.flatten(),
        # "attention_mask": attention_mask.flatten(),
    }
    return encoded_sample


def key_by_problem(samples: List[Dict]):
    grouped_samples = defaultdict(list)
    for sample in samples:
        if "problem" in sample:  # for scored-test-samples.json
            grouped_samples[sample["problem"]].append(sample)
        else:  # for prm800k jsonls
            grouped_samples[sample["question"]["problem"]].append(sample)
    return grouped_samples


def get_answer(sample: Sample) -> Optional[str]:
    return sample.get("answer", sample.get("given_answer", None))


def pad_with_nones(samples, num_total_solution_samples_per_problem):
    nones = [None] * (
        num_total_solution_samples_per_problem - len(samples)
    )  # pad with None
    samples = samples + nones
    return samples


def get_n_samples_considering_failures(
    solution_samples, num_total_solution_samples_per_problem, n, verbose=False
):
    solution_samples = pad_with_nones(
        solution_samples, num_total_solution_samples_per_problem
    )
    random.shuffle(solution_samples)
    subsamples = list(solution_samples[:n])

    subsamples = [x for x in subsamples if x is not None]
    subsamples = [x for x in subsamples if get_answer(x) is not None]

    if verbose:
        print("len(problem_samples) =", len(solution_samples))
        print("len(subsamples) =", len(subsamples))
        print("len(subsamples_not_none) =", len(subsamples))
        print("len(subsamples_with_answer) =", len(subsamples))

    return subsamples


def choose_sample_by_method(samples: List[Sample], method: str) -> Optional[Sample]:
    if len(samples) == 0:
        return None
    if method == "majority_voting":
        ans2num_repr = {}  # (ans_num, repr_sample)
        for sample in samples:
            if ans2num_repr.get(sample["answer"], None) is None:
                ans2num_repr[sample["answer"]] = [1, sample]
            ans2num_repr[sample["answer"]][0] += 1
        majority_ans = max(ans2num_repr, key=lambda x: ans2num_repr[x][0])
        majority_sample = ans2num_repr[majority_ans][1]
        return majority_sample

    return max(samples, key=lambda x: x[method])


def eval_best_of_n_on_rated_problem_solution_samples(
    rated_problem_solution_samples,
    num_trials,
    num_samples_per_problem,
    ns=None,
    metrics=[
        "majority_voting",
        "positive_probs_product",
        "positive_probs_minimum",
        "non_negative_probs_product",
        "non_negative_probs_minimum",
    ],
    best_of_n_results_jsonl_path=best_of_n_results_jsonl_path,
    model_name_or_path=model_path,
    verbose=False,
    debug_for={},
):
    # parameter preparation
    if debug_for is None:
        debug_for = {}

    if ns is None or ns == []:
        ns = []
        n = 2
        while n <= num_samples_per_problem:
            ns.append(n)
            n = n * 2

    num_problems = len(rated_problem_solution_samples)

    max_len_of_metric_name = max([len(metric) for metric in metrics])

    # for : problem / trial -> n -> metric
    # trial 要记录所有的 n，因此在 n 之前
    # n_samples 有随机性，需要 metric 在 n 之后，共用同 n 个 subsamples
    # trial 最后要取平均，可记录 [metric][trial, n_idx]

    all_trial_pass_rates_by_metric = {}
    for metric in metrics:
        all_trial_pass_rates_by_metric[metric] = []

    # main loop
    for trial in range(num_trials):
        # init pass_rates_by_metric
        pass_rates_by_metric = {}
        for metric in metrics:
            pass_rates_by_metric[metric] = []

        # continue main loop
        for n in ns:
            if debug_for.get("n"):
                print("n =", n)

            # init num_correct_by_metric
            num_correct_by_metric = {}
            for metric in metrics:
                num_correct_by_metric[metric] = 0

            # ipdb.set_trace()

            # continue main loop
            for problem_sample in rated_problem_solution_samples:
                n_solution_samples = get_n_samples_considering_failures(
                    problem_sample["solutions"], num_samples_per_problem, n
                )  # num_samples_per_problem != num_total_solution_samples_per_problem !!!

                if debug_for.get("solution_sample"):
                    num_solution_samples = len(n_solution_samples)
                    if num_solution_samples > 0:
                        print("n_solution_samples[0]", n_solution_samples[0])

                for metric in metrics:
                    choice = choose_sample_by_method(n_solution_samples, metric)
                    if choice is not None and choice["is_correct"]:
                        num_correct_by_metric[metric] += 1

            # calculate n-wise pass rate
            for metric in metrics:
                pass_rates_by_metric[metric].append(
                    num_correct_by_metric[metric] / num_problems
                )  # pass_rate = [metric][n_idx]

        print(f"Trial {trial + 1}:")
        for metric in metrics:
            print(f"{metric:>{max_len_of_metric_name}}, {pass_rates_by_metric[metric]}")

        # trial-wise
        for metric in metrics:
            all_trial_pass_rates_by_metric[metric].append(
                pass_rates_by_metric[metric]
            )  # all_trial_pass_rates_by_metric = [metric][trial][n_idx]
            if debug_for.get("all_trial_pass_rates"):
                print(all_trial_pass_rates_by_metric[metric])

    print("Results:")
    print("\tns  :", ns)

    # init
    means_by_metric = {}
    stds_by_metric = {}

    # calculate trial-wise mean and standard deviation
    for metric in metrics:
        if debug_for.get("all_trial_pass_rates"):
            print(all_trial_pass_rates_by_metric[metric])

        means_by_metric[metric] = np.mean(
            all_trial_pass_rates_by_metric[metric], axis=0
        ).tolist()
        stds_by_metric[metric] = np.std(
            all_trial_pass_rates_by_metric[metric], axis=0
        ).tolist()

        print(metric)
        print("\tMean:", means_by_metric[metric])
        print("\tStd :", means_by_metric[metric])

    best_of_n_results = []
    # write to jsonl
    if best_of_n_results_jsonl_path is not None:
        with open(best_of_n_results_jsonl_path, "a") as writer:
            for metric in metrics:
                for n_idx, n in enumerate(ns):
                    best_of_n_result = {
                        "model_name_or_path": "model-agnostic"
                        if metric == "majority_voting"
                        else model_name_or_path.split("models/")[-1],
                        "metric": metric,
                        "num_samples_per_problem": n,
                        "mean": means_by_metric[metric][n_idx],
                        "std": stds_by_metric[metric][n_idx],
                        "num_trials": num_trials,
                        "pass_rates": all_trial_pass_rates_by_metric[metric],
                        "time_stamp": datetime.datetime.now().strftime(
                            "UTC+8 %Y-%m-%d %H:%M:%S %f"
                        ),
                    }
                    best_of_n_results.append(best_of_n_result)
                    writer.write(orjson.dumps(best_of_n_result).decode() + "\n")

    return best_of_n_results


def rate_sample(sample, verbose=False):
    """
    Returns: sample with positive_probs_product, positive_probs_minimum, non_negative_probs_product, non_negative_probs_minimum
    """
    try:
        positive_probs = [
            rating_str2prob["1"] for rating_str2prob in sample["rating_probs"]
        ]
        positive_probs_product = reduce(lambda x, y: x * y, positive_probs)
        positive_probs_minimum = min(positive_probs)

        neutral_probs = [
            rating_str2prob["0"] for rating_str2prob in sample["rating_probs"]
        ]

        non_negative_probs = [p + n for p, n in zip(positive_probs, neutral_probs)]
        non_negative_probs_product = reduce(lambda x, y: x * y, non_negative_probs)
        non_negative_probs_minimum = min(non_negative_probs)

        sample["positive_probs_product"] = positive_probs_product
        sample["positive_probs_minimum"] = positive_probs_minimum
        sample["non_negative_probs_product"] = non_negative_probs_product
        sample["non_negative_probs_minimum"] = non_negative_probs_minimum
    except Exception as e:
        print(e)
        print(sample["rating_probs"])
        ipdb.set_trace()

    if verbose:
        print("prm_score =", sample.get("prm_score"))
        print()
        for rating2prob_step in sample["rating_probs"]:
            print(rating2prob_step)
            print(sum(list(rating2prob_step.values())))
        print()
        print("positive_probs_product =", positive_probs_product)
        print("positive_probs_minimum =", positive_probs_minimum)
        print()
        print("non_negative_probs_product =", non_negative_probs_product)
        print("non_negative_probs_minimum =", non_negative_probs_minimum)

    # return positive_probs_product, positive_probs_minimum, non_negative_probs_product, non_negative_probs_minimum
    return sample


def is_zero3_parameters(model_name_or_path):
    return re.match(r".*/(step|epoch)_[0-9]+$", model_name_or_path) is not None


def extract_step_or_epoch_num(path):
    result = re.search(r".*?(?:step|epoch)_([0-9]+)", path)
    if result is None:
        return None
    return int(result.group(1))


def rate_n_samples(
    model_name_or_path=model_path,
    problem_solution_hierarchical_samples_path=gpt4_generated_problem_solution_hierarchical_samples_path_wo_basename
    + ".pkl",
    num_solution_samples_to_rate_per_problem=num_solution_samples_to_rate_per_problem,
    rated_problem_solution_hierarchical_samples_path=None,
    lib="vllm",
    seed=None,
    debug_for={},
):
    # prepare params
    if debug_for is None:
        debug_for = {}

    if is_zero3_parameters(model_name_or_path) and lib == "vllm":
        # preprocess ZeRO-3 model weights into hf model weights for vllm
        print(f"ZeRO-3 model weights detected at {model_name_or_path}.")

        hf_fp16_model_name_or_path = model_name_or_path + "_fp16_hf"

        if not os.path.exists(hf_fp16_model_name_or_path):
            model_dirpath = os.path.dirname(model_name_or_path)

            # run zero_to_fp32.py
            pytorch_model_filename = "pytorch_model.bin"
            pytorch_model_filepath = os.path.join(
                model_name_or_path, pytorch_model_filename
            )

            if not os.path.exists(pytorch_model_filepath):
                zero_to_fp32_result = subprocess.run(
                    [
                        python_path,
                        "zero_to_fp32.py",
                        model_name_or_path,
                        pytorch_model_filepath,
                    ],
                    cwd=model_path,
                )
                if zero_to_fp32_result.returncode != 0:
                    raise RuntimeError(
                        f"zero_to_fp32.py failed with returncode {zero_to_fp32_result.returncode}"
                    )

            # load pytorch_model.bin and save it as hf model
            model_config = transformers.AutoConfig.from_pretrained(model_dirpath)
            temp_model = transformers.AutoModelForCausalLM.from_pretrained(
                os.path.join(model_name_or_path, pytorch_model_filename),
                config=model_config,
                torch_dtype=torch.float16,  # fp16 to accelerate but try to avoid positional encoding clash
                low_cpu_mem_usage=True,  # memory and time-
            )

            temp_model.save_pretrained(hf_fp16_model_name_or_path)

            temp_model = None  # release the memory

        print(
            f"ZeRO-3 model weights converted to HF model weights at {hf_fp16_model_name_or_path}"
        )
        model_name_or_path = hf_fp16_model_name_or_path

    # must be after checking is_zero3_parameters
    if rated_problem_solution_hierarchical_samples_path == "default":
        rel_model_name_or_path = model_name_or_path.split("models/")[-1]

        rated_problem_solution_hierarchical_samples_path = os.path.join(
            rated_gpt4_generated_problem_solution_hierarchical_samples_dirpath,
            rel_model_name_or_path,
            f"{num_solution_samples_to_rate_per_problem}-samples-per-problem" + ".pkl",
        )

    # scoop if rated_problem_solution_hierarchical_samples_path exists
    if os.path.exists(rated_problem_solution_hierarchical_samples_path):
        print(f"{rated_problem_solution_hierarchical_samples_path} exists.")
        rated_problem_solution_hierarchical_samples = load_pickle(
            rated_problem_solution_hierarchical_samples_path
        )
        return rated_problem_solution_hierarchical_samples

    if lib == "vllm":
        vllm_outputs_path = os.path.join(project_root, "tmp/vllm-outputs.pkl")
        if not debug_for.get("resume_vllm_outputs"):
            # load problem_solution_hierarchical_samples
            print(f"Loading {problem_solution_hierarchical_samples_path}...")
            problem_solution_hierarchical_samples = load_pickle(
                problem_solution_hierarchical_samples_path
            )
            print(f"Loaded")

            # extract n valid subsamples
            print(
                f"Extracting {num_solution_samples_to_rate_per_problem} subsamples..."
            )
            if seed is not None:
                random.seed(seed)
                print(f"random.seed({seed})")
            for problem_sample in problem_solution_hierarchical_samples:
                problem_sample["solutions"] = get_n_samples_considering_failures(
                    problem_sample["solutions"],
                    num_total_solution_samples_per_problem,
                    num_solution_samples_to_rate_per_problem,
                )
            print(f"Extracted")
            # construct prompts
            prompts = problem_samples2prompts(problem_solution_hierarchical_samples)

            if debug_for.get("prompts"):
                print(prompts[0])

            llm = get_vllm(model_name_or_path=model_path)
            outputs = prm800k_vllm_inference(
                llm, generation_config=generation_config, prompts=prompts
            )  # 13:28
        else:
            problem_solution_hierarchical_samples_path = os.path.join(
                project_root, "tmp/problem_solution_hierarchical_samples.pkl"
            )
            print(f"Loading {problem_solution_hierarchical_samples_path}...")
            problem_solution_hierarchical_samples = load_pickle(
                problem_solution_hierarchical_samples_path
            )
            print(f"Loaded")
            print(f"Resuming vLLM outputs from {vllm_outputs_path}")
            outputs = load_pickle(vllm_outputs_path)
            print(f"Resumed")

        if debug_for.get("save_vllm_outputs"):
            save_pickle(
                problem_solution_hierarchical_samples_path,
            )
            save_pickle(outputs, vllm_outputs_path)

        # tokenizer = get_complete_tokenizer()
        rating2prob_list = vllm_outputs2rating2prob_list(outputs, tokenizer)  # 14.7s
    else:
        raise NotImplementedError()

    # rating2prob_list done

    num_step_so_far = 0
    for problem_sample in problem_solution_hierarchical_samples:
        for solution_sample in problem_sample["solutions"]:
            solution_rating2prob_list = solution_sample["rating_probs"]
            solution_step_num = len(solution_rating2prob_list)
            for step_idx in range(solution_step_num):
                try:
                    solution_rating2prob_list[step_idx] = rating2prob_list[
                        num_step_so_far + step_idx
                    ]
                except IndexError:
                    print(f"num_step_so_far = {num_step_so_far}")
                    print(f"step_idx = {step_idx}")
                    print(f"len(rating2prob_list) = {len(rating2prob_list)}")
                    ipdb.set_trace()
            num_step_so_far += solution_step_num
            # print(f"num_step_so_far = {num_step_so_far}")
            rate_sample(solution_sample)

    # problem_solution_hierarchical_samples rated

    if debug_for.get("rated_samples"):
        print(
            "problem_solution_hierarchical_samples[0]['solutions'][0]:",
            problem_solution_hierarchical_samples[0]["solutions"][0],
        )

    if rated_problem_solution_hierarchical_samples_path is not None:
        save_pickle(
            problem_solution_hierarchical_samples,
            rated_problem_solution_hierarchical_samples_path,
        )

    return problem_solution_hierarchical_samples  # rated


def eval_model_with_best_of_n(
    model_name_or_path=model_path,
    problem_solution_hierarchical_samples_path=gpt4_generated_problem_solution_hierarchical_samples_path_wo_basename
    + ".pkl",
    num_solution_samples_to_rate_per_problem=num_solution_samples_to_rate_per_problem,
    best_of_n_results_jsonl_path=best_of_n_results_jsonl_path,
    metrics=all_metrics,
    num_trials=num_trials,
    debug_for={},
):
    # eval

    rated_problem_solution_hierarchical_samples = rate_n_samples(
        model_name_or_path=model_path,
        problem_solution_hierarchical_samples_path=problem_solution_hierarchical_samples_path,
        num_solution_samples_to_rate_per_problem=num_solution_samples_to_rate_per_problem,
        rated_problem_solution_hierarchical_samples_path="default",
        lib="vllm",
        debug_for=debug_for,
    )

    _ = eval_best_of_n_on_rated_problem_solution_samples(
        rated_problem_solution_samples=rated_problem_solution_hierarchical_samples,
        num_trials=num_trials,
        num_samples_per_problem=num_solution_samples_to_rate_per_problem,
        ns=None,
        metrics=metrics,
        best_of_n_results_jsonl_path=best_of_n_results_jsonl_path,
        model_name_or_path=model_path,
        verbose=False,
        debug_for=debug_for,
    )


def get_rating_objs(tokenizer, rating2word=rating2word, verbose=False):
    """
    Returns: rating2word, rating_words, rating_token_ids, token_id2rating_str
    """

    rating_words = list(rating2word.values())
    rating_token_ids = tokenizer(
        rating_words, add_special_tokens=False
    ).input_ids  # [[6374], [8178], [21104]]
    rating_token_ids = set(
        [token_id[0] for token_id in rating_token_ids]
    )  # [6374, 8178, 21104]
    token_id2rating_str = {
        tokenizer(word, add_special_tokens=False).input_ids[0]: str(rating)
        for rating, word in rating2word.items()
    }  # {6374: '1', 8178: '-1', 21104: '0'}

    if verbose:
        print("rating2word =", rating2word)
        print("rating_words =", rating_words)
        print("rating_token_ids =", rating_token_ids)
        print("token_id2rating_str =", token_id2rating_str)

    return rating2word, rating_words, rating_token_ids, token_id2rating_str


# tokenizer

# LLaMA


def get_hf_model(model_name_or_path=model_path, **kwargs):
    """fp16, low_cpu_mem_usage"""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        **kwargs,
    )
    print(type(model))
    print(model.config)
    return model


def check_tokenizer(tokenizer, head=5):
    # if "accelerator" in globals():
    #     print = accelerator.print
    # else:
    #     print = print
    head_tokens = tokenizer.decode(list(range(head)))
    print(f"head_tokens = {head_tokens}")
    print(f"tokenizer.vocab_size = {tokenizer.vocab_size}")
    print(f"len(tokenizer.vocab) = {len(tokenizer.vocab)}")
    print(f"tokenizer.special_tokens_map = {tokenizer.special_tokens_map}")


# no default pad token for llama!
# here we add all special tokens again, because the default ones are not in the special_tokens_map
def complete_four_special_tokens(tokenizer):
    check_tokenizer(tokenizer)

    num_added_tokens = tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
    )
    assert num_added_tokens in [
        0,
        1,
    ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    # tokenizer.vocab_size += num_added_tokens # AttributeError: can't set attribute 'vocab_size'

    print(f"num_added_tokens = {num_added_tokens}")
    check_tokenizer(tokenizer)

    return tokenizer


def get_complete_tokenizer(tokenizer_name_or_path=tokenizer_name_or_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, model_max_length=model_max_length
        )
        tokenizer = complete_four_special_tokens(tokenizer)

    print(tokenizer)

    return tokenizer


def preprocess(tokenizer, prompts, **kwargs):
    inputs = tokenizer(prompts, return_tensors="pt", **kwargs)
    return inputs


# rating


def problem_steps2solution_till_step_list(problem, steps):
    solution_till_step = []
    solution_so_far = problem
    for step in steps:
        solution_so_far += "\n" + step.strip()
        solution_till_step.append(solution_so_far)
    return solution_till_step


def problem_samples2prompts(problem_samples, verbose=False):
    prompts = []

    for problem_sample in problem_samples:
        problem = problem_sample["problem"]

        for solution_sample in problem_sample["solutions"]:
            steps = solution_sample["steps"]
            prompts.extend(problem_steps2solution_till_step_list(problem, steps))

        if verbose:
            print(f"len(prompts) = {len(prompts)}")

    print(f"len(prompts) = {len(prompts)}")
    print(f"prompts[0] = {prompts[0]}")

    return prompts


def problem_solution2input_ids_list(tokenizer, problem, steps):
    solution_input_ids_list = []

    problem_input_ids = tokenizer(
        problem + "\n",
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )["input_ids"][0]

    solution_input_ids = problem_input_ids

    for step in steps:
        step = step.strip()

        step_input_ids = tokenizer(
            "\n" + step + "\n",
            return_tensors="pt",
            padding=False,
            truncation=False,
            # add_special_tokens=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"][0]

        step_input_ids = step_input_ids[2:]  # remove "\n" at beginning
        solution_input_ids = torch.cat((solution_input_ids, step_input_ids), dim=-1)[
            :-1
        ]
        solution_input_ids_list.append(solution_input_ids)

    return solution_input_ids_list


# vllm


def get_vllm(model_name_or_path=model_path):
    # global llm

    # if llm is None:
    llm = vllm.LLM(
        model=model_path,
        tokenizer=tokenizer_name_or_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        seed=0,
    )

    return llm


def prm800k_vllm_inference(
    llm, generation_config, prompts=None, prompt_input_ids_list=None
):
    sampling_params = vllm.SamplingParams(**generation_config)

    # new_sample = sample.copy()

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    outputs = llm.generate(
        prompts=prompts,
        prompt_token_ids=prompt_input_ids_list,
        sampling_params=sampling_params,
    )

    print(f"len(outputs) = {len(outputs)}")
    print(f"outputs[0] = {outputs[0]}")

    return outputs


def vllm_outputs2rating2prob_list(
    outputs, tokenizer=None, rating2word=rating2word, verbose=False
):
    rating2prob_list = []

    for idx, output in enumerate(outputs):
        # _, _, rating_token_ids, token_id2rating_str = get_rating_objs(
        #     tokenizer, rating2word
        # )
        token_id2rating_str = {6374: "1", 8178: "-1", 21104: "0"}
        rating_token_ids = list(token_id2rating_str.keys())

        token_id2logprob = output.outputs[0].logprobs[0]

        top_token_ids = list(token_id2logprob.keys())

        if not all([token_id in top_token_ids for token_id in rating_token_ids]):
            print(f"[ERROR] top_token_ids out of top-{top_k} tokens!")
            print("idx =", idx)
            print("top_ids =", top_token_ids)
            print("logprobs =", token_id2logprob)

        for token_id in top_token_ids:
            if token_id not in rating_token_ids:
                del token_id2logprob[token_id]
        assert set(token_id2logprob.keys()) == set(rating_token_ids), token_id2logprob

        logprobs = list(token_id2logprob.values())
        probs = np.exp(logprobs)
        sum_probs = sum(probs)

        norm_probs = probs / sum_probs

        rating_strs = [
            token_id2rating_str[token_id] for token_id in token_id2rating_str.keys()
        ]
        rating_probs = {
            rating_str: norm_prob
            for rating_str, norm_prob in zip(rating_strs, norm_probs)
        }
        rating2prob_list.append(rating_probs)
        # break

        if verbose:
            print(output)
            print("logprobs =", token_id2logprob)
            print("probs =", probs)
            print("sum(probs) =", sum_probs)
            print("norm_probs =", norm_probs)
            print("rating_probs =", rating_probs)

    print("len(rating2prob_list) =", len(rating2prob_list))
    print("rating2prob_list[0] =", rating2prob_list[0])

    return rating2prob_list
