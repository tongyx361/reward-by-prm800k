#!/usr/bin/env python
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
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import bitsandbytes as bnb
import blobfile as bf
import datasets
import evaluate
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import orjson
# import prepare_dataset
import regex as re
import torch
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import vllm
import wandb

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
    # "roc_auc",
]

default_max_seq_len = 1024

# print


def local_main_print(*args, verbose: bool = False, **kwargs):
    if verbose:
        print(f"[DEBUG] os.environ['LOCAL_RANK'] = {os.environ['LOCAL_RANK']}")
    if os.environ["LOCAL_RANK"] == "0":
        print(*args, **kwargs)


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
        file = logging.FileHandler(log_file_path, encoding="utf-8")
        file.setLevel(level)
        file.setFormatter(formatter)
        logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    print(logger)

    return logger


# environmetn variables

data_root = os.environ.get("DATA_ROOT", "/data/users/zhangjunlei/tyx")
hf_home = os.path.join(data_root, ".cache/huggingface")

os.environ.update(
    {
        "DATA_ROOT": data_root,
        "HF_HOME": hf_home,
        "TRANSFORMERS_CACHE": os.path.join(hf_home, "hub"),
        "HF_DATASETS_CACHE": os.path.join(hf_home, "datasets"),
        "HF_MODULES_CACHE": os.path.join(hf_home, "modules"),
        "HF_METRICS_CACHE": os.path.join(hf_home, "metrics"),
    }
)

project_root = os.path.join(data_root, "reward-by-prm800k")
models_root = os.path.join(project_root, "models")

default_model_name = "meta-llama/Llama-2-7b-hf"
default_7b_model_path = os.path.join(
    hf_home,
    "hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9",
)

default_13b_model_path = os.path.join(
    hf_home,
    "hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936",
)

default_tokenizer_path = os.path.join(
    data_root,
    ".cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470",
)
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


# python environment

if "zhangjunlei" in data_root:
    conda_env_path = "/data/users/zhangjunlei/anaconda3/envs/open-instruct"
elif "tongyx361" in data_root:
    conda_env_path = "/data/tongyx361/miniconda3/envs/nlp"


python_path = os.path.join(conda_env_path, "bin/python")

# PRM800K

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


def get_prm800k_all_train_samples(
    prm800k_phase_train_jsonl_paths=prm800k_phase_train_jsonl_paths,
):
    prm800k_all_train_samples = []
    for idx, data_files_train in enumerate(prm800k_phase_train_jsonl_paths):
        prm800k_phase_train = load_jsonl(data_files_train)
        prm800k_all_train_samples.extend(prm800k_phase_train)
    prm800k_all_train_samples = list(
        filter(pick_prm800k_samples, prm800k_all_train_samples)
    )
    print(f"len(prm800k_all_train_samples) = {len(prm800k_all_train_samples)}")
    print(f"example: {random.choice(prm800k_all_train_samples)}")

    return prm800k_all_train_samples


# singelton

llm = None
tokenizer = None

# model

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


def get_data_module_from_encoded_datasets(encoded_datasets_path):
    encoded_datasets = datasets.load_from_disk(encoded_datasets_path)
    data_module = {
        "train_dataset": encoded_datasets["train"],
        "eval_dataset": encoded_datasets["validation"],
    }
    return data_module


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    # mask the prompt part for avoiding loss
    labels[
        :, : tokenized_prompt.input_ids.shape[1]
    ] = -100  # tokenized_prompt.input_ids 的形状是 (batch_size, sequence_length)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_problem_step_ratings_format(
    reformatted_sample, tokenizer, max_seq_length, test=False
):
    """
    Here we assume each example has a 'step_ratings' field. Each step_rating is a dict.
    """
    # reformatted_sample = reformat_prm800k_sample(sample)
    step_ratings = reformatted_sample["step_ratings"]
    if len(step_ratings) == 0:
        raise ValueError("step_ratings field is empty.")

    # rating2token = {1: "<pos>", -1: "<neg>", 0: "<neu>"}
    rating2token = {1: "positive", -1: "negative", 0: "neutral"}

    def _concat_step_ratings(step_ratings):
        step_ratings_text = reformatted_sample["problem"] + "\n"
        for step_rating in step_ratings:
            step_ratings_text += (
                step_rating["step"].strip()
                + tokenizer.cls_token
                + rating2token[step_rating["rating"]]
                + "\n"
            )

        return step_ratings_text

    example_text = _concat_step_ratings(step_ratings).strip()  # remove the last \n
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )

    if test is True:
        print("tokenized_example: ", tokenized_example, sep="\n")

    input_ids = tokenized_example.input_ids
    labels = torch.ones_like(input_ids).mul(
        -100
    )  # mask the non-rating part for avoiding loss

    # mask the non-rating part for avoiding loss
    for idx, input_id in enumerate(input_ids[0]):
        if input_id == tokenizer.cls_token_id:
            rating_idx = idx + 1
            labels[0][rating_idx] = input_ids[0][rating_idx]

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = PaddingStrategy.LONGEST
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )

        # truncate
        if self.max_length is not None:
            for feature in features:
                for k, v in feature.items():
                    if len(v) > self.max_length:
                        feature[k] = v[: self.max_length]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


def get_data_collator(
    tokenizer, model=None, padding="longest", max_length=default_max_seq_len
):
    return DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model=model,
        padding=padding,
        max_length=max_length,
    )


# device


def nvidia_smi(verbose=True):
    ns_result = subprocess.run(["nvidia-smi"], capture_output=True)
    output = ns_result.stdout.decode("utf-8")
    if verbose:
        local_main_print(output)
    return output


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
        return [json_loads(line) for line in f.readlines() if line]


# def dump_save(module, data, path, **kwargs):


def load_json(path, data_type=dict):
    if os.path.exists(path):
        f = open(path)
        data = json.load(f)
        f.close()
        if data is None:
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
        if data is None:
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


# debug


def check_time_cost(code_list):
    total_start_time = time.time()
    for code in code_list:
        start_time = time.time()
        exec(code)
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"{time_cost} seconds to execute `{code}`")
    total_end_time = time.time()
    total_time_cost = total_end_time - total_start_time
    print(f"{total_time_cost} seconds to execute all codes")


# metrics


def evaluate_clf_metrics_compute(
    metrics, flat_preds, flat_refs, labels, token_probs_list=None
):
    eval_metrics = {}
    # ROC
    # eval_metrics["roc"] = wandb.plot.roc_curve(
    #     y_true=flat_refs, y_probas=token_rating_probs_list
    # )

    # nums
    pred_nums = defaultdict(int)
    ref_nums = defaultdict(int)
    for pred, ref in zip(flat_preds, flat_refs):
        pred_nums[pred] += 1
        ref_nums[ref] += 1

    # ::DONE:: shorten the following keys
    for key, value in pred_nums.items():
        eval_metrics[f"pred_{key}_num"] = value

    for key, value in ref_nums.items():
        eval_metrics[f"ref_{key}_num"] = value

    if isinstance(metrics, evaluate.CombinedEvaluations):
        metrics = metrics.evaluation_modules
    for metric in metrics:
        if metric.name == "rocauc":  # not roc_auc!
            if token_probs_list is not None:
                for average in MULTICLASS_AVERAGINGS:
                    average_roc_auc = metric.compute(
                        references=flat_refs,
                        prediction_scores=token_probs_list,
                        average=average,
                        multi_class="ovr",
                    )[
                        "roc_auc"
                    ]  # not metric.name!
                    eval_metrics[f"roc_auc_{average}"] = average_roc_auc
                class_roc_aucs = metric.compute(
                    references=flat_refs,
                    prediction_scores=token_probs_list,
                    labels=labels,
                    average=None,
                    multi_class="ovr",
                )["roc_auc"]
                eval_metrics.update(
                    {
                        f"roc_auc_{labels[idx]}": value
                        for idx, value in enumerate(class_roc_aucs)
                    }
                )
            continue

        if metric.name not in MUL_CLF_METRIC_NAMES:
            eval_metrics.update(
                metric.compute(predictions=flat_preds, references=flat_refs)
            )
        else:  # metric.name in utils.MUL_CLF_METRIC_NAMES:
            for average in MULTICLASS_AVERAGINGS:
                eval_metrics.update(
                    {
                        f"{metric.name}_{average}": metric.compute(
                            predictions=flat_preds,
                            references=flat_refs,
                            average=average,
                        )[metric.name]
                    }
                )
            class_metrics = metric.compute(
                predictions=flat_preds,
                references=flat_refs,
                average=None,
                labels=labels,
            )[metric.name]
            eval_metrics.update(
                {
                    f"{metric.name}_{labels[idx]}": value
                    for idx, value in enumerate(class_metrics)
                }
            )

    return eval_metrics


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


# lora


def find_all_linear_names(bits: int, model):
    assert bits in [4, 8], "bits must be 4 or 8"

    cls = (
        bnb.nn.Linear4bit
        if bits == 4
        else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# visualize


def visualize_dict_value_lengths_distribuition(data_dict):
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

    visualize_dict_value_lengths_distribuition(data_dict)


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


# PRM800K evalution

rating_token_id2label = {8178: -1, 21104: 0, 6374: 1}
rating_token_ids = list(rating_token_id2label.keys())
rating_labels = list(rating_token_id2label.values())


def prm800k_preprocess_logits_for_metrics(logits, labels):
    rating_logits = logits[:, :, rating_token_ids]

    # eval_prediction = {
    #     "predictions": rating_logits,
    #     "label_ids": labels,
    # }

    return rating_logits


def prm800k_compute_metrics(eval_prediction):
    metric_results = {}
    IGNORE_IDX = -100

    # Load metrics
    clf_metrics = [evaluate.load(name) for name in CLF_METRIC_NAMES] + [
        evaluate.load("roc_auc", "multiclass")
    ]

    # Load data
    raw_seq_rating_logits_list, raw_seq_label_ids = (
        eval_prediction.predictions,
        eval_prediction.label_ids,
    )

    local_main_print(raw_seq_rating_logits_list.shape)

    # left shift labels by 1
    shifted_seq_rating_logits_list = raw_seq_rating_logits_list[:, :-1, :]
    shifted_seq_label_ids = raw_seq_label_ids[:, 1:]

    token_rating_logits_list = []
    flat_label_ids = []

    for seq_rating_logits_list, seq_label_ids in zip(
        shifted_seq_rating_logits_list, shifted_seq_label_ids
    ):
        for token_rating_logits, label_id in zip(seq_rating_logits_list, seq_label_ids):
            if label_id != IGNORE_IDX:
                token_rating_logits_list.append(token_rating_logits)
                flat_label_ids.append(label_id)

    # token_rating_logits_list = torch.tensor(token_rating_logits_list)
    token_rating_logits_list = torch.tensor(np.array(token_rating_logits_list))
    flat_preds = []
    flat_refs = []

    token_rating_probs_list = torch.softmax(token_rating_logits_list, dim=-1)
    rating_pred_idxs = token_rating_logits_list.argmax(dim=-1)  # greedy

    for pred_idx, label_id in zip(rating_pred_idxs, flat_label_ids):
        flat_preds.append(rating_labels[pred_idx])
        flat_refs.append(rating_token_id2label[label_id])

    # nums
    pred_nums = defaultdict(int)
    ref_nums = defaultdict(int)
    for pred, ref in zip(flat_preds, flat_refs):
        pred_nums[pred] += 1
        ref_nums[ref] += 1

    # ::DONE:: shorten the following keys
    for key, value in pred_nums.items():
        metric_results[f"pred_{key}_num"] = value

    for key, value in ref_nums.items():
        metric_results[f"ref_{key}_num"] = value

    for metric in clf_metrics:
        if metric.name == "rocauc":  # not roc_auc!
            for average in MULTICLASS_AVERAGINGS:
                average_roc_auc = metric.compute(
                    references=flat_refs,
                    prediction_scores=token_rating_probs_list,
                    average=average,
                    multi_class="ovr",
                )[
                    "roc_auc"
                ]  # not metric.name!
                metric_results[f"roc_auc_{average}"] = average_roc_auc
            class_roc_aucs = metric.compute(
                references=flat_refs,
                prediction_scores=token_rating_probs_list,
                labels=rating_labels,
                average=None,
                multi_class="ovr",
            )["roc_auc"]
            metric_results.update(
                {
                    f"roc_auc_{rating_labels[idx]}": value
                    for idx, value in enumerate(class_roc_aucs)
                }
            )
            continue

        if metric.name not in MUL_CLF_METRIC_NAMES:
            metric_results.update(
                metric.compute(predictions=flat_preds, references=flat_refs)
            )
        else:  # metric.name in utils.MUL_CLF_METRIC_NAMES:
            for average in MULTICLASS_AVERAGINGS:
                metric_results.update(
                    {
                        f"{metric.name}_{average}": metric.compute(
                            predictions=flat_preds,
                            references=flat_refs,
                            average=average,
                        )[metric.name]
                    }
                )
            class_metrics = metric.compute(
                predictions=flat_preds,
                references=flat_refs,
                average=None,
                labels=rating_labels,
            )[metric.name]
            metric_results.update(
                {
                    f"{metric.name}_{rating_labels[idx]}": value
                    for idx, value in enumerate(class_metrics)
                }
            )
    return metric_results


# PRM800K data processing


def prm800k_extract_synthesized_analysis(
    synthesized_analysis: str, query_type: str = "sar", debug={}
):
    steps = []
    ratings = []
    analyses = []
    prompt_with_analysis = "## Step-Analysis-Rating\n"
    step_rating_analysis_section = synthesized_analysis.split(prompt_with_analysis)[-1]
    if query_type == "sar":
        pattern = r'Step \d+: (?:""")?(.+?)(?:""")? Analysis: (.+?) Rating: (-1|0|1)'
    elif query_type == "sra":
        pattern = r'Step \d+: (?:""")?(.+?)(?:""")? Rating: (-1|0|1) Analysis: (.+?)'
    else:
        raise ValueError(f"query_type `{query_type}` is not valid")

    search_results = re.findall(pattern, step_rating_analysis_section, flags=re.DOTALL)

    if debug.get("search"):
        print(search_results)
    for idx, step_rating_analysis in enumerate(search_results):
        if query_type == "sar":
            step, analysis, rating = step_rating_analysis
        elif query_type == "sra":
            step, rating, analysis = step_rating_analysis
        else:
            raise ValueError(f"query_type `{query_type}` is not valid")

        step = step.strip()
        analysis = analysis.strip()
        rating = rating.strip()

        assert rating in ("-1", "0", "1"), f"rating `{rating}` is not valid"
        # assert (
        #     rating in analysis.split(",")[-1]
        # ), f"rating {rating} is not in analysis {analysis}"
        steps.append(step)
        ratings.append(rating)
        analyses.append(analysis)

    return steps, ratings, analyses


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
    model_name_or_path=default_7b_model_path,
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
    model_name_or_path=default_7b_model_path,
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
                    cwd=default_7b_model_path,
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
            print("Loaded")

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
            print("Extracted")
            # construct prompts
            prompts = problem_samples2prompts(problem_solution_hierarchical_samples)

            if debug_for.get("prompts"):
                print(prompts[0])

            llm = get_vllm(model_name_or_path=default_7b_model_path)
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
            print("Loaded")
            print(f"Resuming vLLM outputs from {vllm_outputs_path}")
            outputs = load_pickle(vllm_outputs_path)
            print("Resumed")

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
    model_name_or_path=default_7b_model_path,
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
        model_name_or_path=default_7b_model_path,
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
        model_name_or_path=default_7b_model_path,
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


def get_hf_model(model_name_or_path=default_7b_model_path, **kwargs):
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


def add_pad_token(tokenizer, pad_token="<pad>"):
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": pad_token})

    assert num_added_tokens == 1, "tokenizer.pad_token already exists"

    local_main_print(f"tokenizer.pad_token = {tokenizer.pad_token}")

    return tokenizer


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


def get_complete_tokenizer(tokenizer_name_or_path=default_tokenizer_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, model_max_length=model_max_length
        )
        tokenizer = complete_four_special_tokens(tokenizer)

    print(tokenizer)
    print(tokenizer.special_tokens_map)
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


def get_vllm(model_name_or_path=default_7b_model_path):
    # global llm

    # if llm is None:
    llm = vllm.LLM(
        model=default_7b_model_path,
        tokenizer=default_tokenizer_path,
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
    return rating2prob_list
    return rating2prob_list
