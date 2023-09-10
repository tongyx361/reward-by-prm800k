#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="my-awesome-project",
        help="The name of the project to log in the experiment tracker.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--use_lora",
    #     action="store_true",
    #     help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    # )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
        choices=("", "lora", "qlora"),
        help="Which PEFT to use.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--save_merged_lora_model",
        action="store_true",
        help="If passed, will merge the lora modules and save the entire model.",
    )
    # parser.add_argument(
    #     "--use_flash_attn",
    #     action="store_true",
    #     help="If passed, will use flash attention to train the model.",
    # )

    # argument to choose which accelerating library to use
    parser.add_argument(
        "--use_accelerate_lib",
        type=str,
        default="",
        choices=["flash-attn-v1", "flash-attn-v2", "xformers", ""],
        help='"flash-attn-v1", "flash-attn-v2", "xformers"',
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation (validation/test) dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--data_seed", type=int, default=None, help="The seed to use for shuffling."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--epoch_checkpointing",
        action="store_true",
        help="If passed, will save the accelerator states at the end of each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--resume_tracking",
        type=str,
        default=None,
        choices=["allow", "must", ""],
        help="Whether to resume the experiment trackers.",
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default=None,
        help="The run id of the experiment to resume. Only applicable when `--resume_tracking` is passed.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # parser.add_argument(
    #     "--use_cache",
    #     action="store_true",
    #     help="If passed, will use k/v cache to speed up.",
    # )
    parser.add_argument(
        "--prm800k",
        action="store_true",
        help=("finetune with PRM800K dataset"),
    )
    parser.add_argument(
        "--encoded_datasets_name_or_path",
        type=str,
        default=None,
        help=("encoded datasets"),
    )
    parser.add_argument(
        "--sync_cache_flush",
        action="store_true",
        help=("sync cache flush"),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("debug mode"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        help="Whether to run eval before training.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Run an evaluation every X steps",
    )
    parser.add_argument(
        "--save_best_metric",
        type=str,
        help="Save the best model checkpoint by the given metric",
    )
    # parser.add_argument(
    #     "--class_average",
    #     type=str,
    #     choices=["binary"] + MULTICLASS_AVERAGINGS,
    #     default="none",
    #     help="The averaging strategy for classification.",
    # )

    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.encoded_datasets_name_or_path is None
    ):
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json/jsonl file."

    return args


args = parse_args()


import logging
import math
import os
import pathlib
import random
import shutil
import time

# import subprocess
from collections import defaultdict
from functools import partial

import datasets
import evaluate

# import ipdb
import torch
import transformers
import utils
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk  # concatenate_datasets,

# from eval import compute_metrics
from peft import LoraConfig, TaskType, get_peft_model
from prepare_dataset import (  # instead of DataCollatorForSeq2Seq,
    DataCollatorForCausalLM,
    encode_with_messages_format,
    encode_with_problem_step_ratings_format,
    encode_with_prompt_completion_format,
)

# from prepare_dataset import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    OPTForCausalLM,
    SchedulerType,
    get_scheduler,
)
from utils import MUL_CLF_METRIC_NAMES, MULTICLASS_AVERAGINGS

import wandb

# post-processing

args.lr_scheduler_type = SchedulerType(args.lr_scheduler_type)

if args.debug:
    args.resume_tracking = False
# A hacky way to make llama work with flash attention
# Note: Need to call this before importing transformers. (FastChat)
if (
    args.use_accelerate_lib is not None
    and args.use_accelerate_lib != ""
    # and "70b" not in args.model_name_or_path.lower()
):
    if args.use_accelerate_lib == "flash-attn-v2":
        utils.local_main_print("Using flash attention v2")
        from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()
    elif args.use_accelerate_lib == "flash-attn-v1":
        utils.local_main_print("Using flash attention v1")
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()
    elif args.use_accelerate_lib == "xformers":
        utils.local_main_print("Using xformers")
        from llama_xformers_attn_monkey_patch import (
            replace_llama_attn_with_xformers_attn,
        )

        replace_llama_attn_with_xformers_attn()


def save_state_to_disk(output_dirpath, output_root=None):
    if not output_dirpath.startswith("/"):
        if output_root is not None:
            output_dirpath = os.path.join(output_root, output_dirpath)
        else:
            output_dirpath = os.path.join(args.output_dir, output_dirpath)
    logger.info(f"Saving state to {output_dirpath}")
    accelerator.wait_for_everyone()  # needed for syncing
    # RuntimeError: Detected mismatch between collectives on ranks. Rank 3 is running collective: CollectiveFingerPrint(OpType=BARRIER), but Rank 0 is running collective: CollectiveFingerPrint(OpType=ALLGATHER_COALESCED).
    # if accelerator.is_local_main_process:
    accelerator.save_state(output_dirpath)
    accelerator.wait_for_everyone()


def inference_and_compute_metrics(model, eval_dataloader, metrics, steps=None):
    # set model to eval mode
    model.eval()

    # flat_preds = torch.tensor([])
    # flat_refs = torch.tensor([])
    # flat_preds = []
    # flat_refs = []

    # raw results just after inference
    # raw_inputs = defaultdict(list)
    raw_input_ids = []
    raw_labels = []
    # raw_logits = []
    all_seq_rating_logits = []

    if args.prm800k:
        rating_token_id2label = {8178: -1, 21104: 0, 6374: 1}
        rating_token_ids = list(rating_token_id2label.keys())
        labels = list(rating_token_id2label.values())
    else:
        raise NotImplementedError("Only support PRM800K for now.")

    progress = tqdm(
        eval_dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    )

    # samples_seen = 0 # no need since we used `Accelerator.gather_for_metrics` instead of `Accelerator.gather`:
    # epoch_max_seq_len = 0
    for step, batch in enumerate(eval_dataloader):
        accelerator.print(f"step: {step}")
        # We can avoid the following line since we set the accelerator with `device_placement=True`.
        # batch.to(accelerator.device)
        inputs = batch
        with torch.no_grad():
            logits = model(**inputs).logits

        rating_logits = logits[:, :, rating_token_ids]

        # predictions = predictions[:, :-1]  # remove the last token
        # references = references[:, 1:]  # remove the first token
        # reference shift left by 1

        # pad_token_id = 32000  # for LLaMA only
        ignore_idx = -100

        seq_len_dim = 1

        inputs, rating_logits = accelerator.pad_across_processes(
            (inputs, rating_logits),
            dim=seq_len_dim,
            pad_index=ignore_idx,
            pad_first=False,
        )

        # references = accelerator.pad_across_processes(
        #     references, dim=seq_len_dim, pad_index=ignore_idx, pad_first=False
        # )

        # predictions = accelerator.pad_across_processes(
        #     predictions, dim=seq_len_dim, pad_index=ignore_idx, pad_first=False
        # )

        # contiguous
        for tensor in inputs.values():
            tensor = tensor.contiguous()
        rating_logits = rating_logits.contiguous()

        # Note: tensor.shape should be aligned before gathering
        inputs, rating_logits = accelerator.gather_for_metrics((inputs, rating_logits))

        if args.debug:
            accelerator.print(f"inputs = {inputs}")
            accelerator.print(f"rating_logits = {rating_logits}")
            # raise NotImplementedError("Debug")

        # logits, references = accelerator.gather_for_metrics(
        #     (outputs.logits, batch["labels"])
        # )

        # The following snippet can be avoided since we used `Accelerator.gather_for_metrics` instead of `Accelerator.gather`:
        # # First we check if it's a distributed system
        # if accelerator.use_distributed:
        #     # Then see if we're on the last batch of our eval dataloader
        #     if step == len(eval_dataloader) - 1:
        #         # Last batch needs to be truncated on distributed systems as it contains additional samples
        #         predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
        #         references = references[: len(eval_dataloader.dataset) - samples_seen]
        #     else:
        #         # Otherwise we add the number of samples seen
        #         samples_seen += references.shape[0]

        # if args.debug:
        #     accelerator.print(f"predictions = {predictions}")
        #     accelerator.print(f"references = {references}")
        #     ipdb.set_trace()

        # gathered

        # epoch_max_seq_len = max(epoch_max_seq_len, logits.shape[seq_len_dim])

        # for key, value in inputs.items():
        #     raw_inputs[key].extend(value.tolist())

        # 8.271041870117188 s
        start_time = time.time()
        raw_input_ids.extend(inputs["input_ids"].tolist())
        raw_labels.extend(inputs["labels"].tolist())
        # raw_logits.extend(logits.tolist())
        all_seq_rating_logits.extend(rating_logits.tolist())
        accelerator.print(f"extend+tolist() time: {time.time() - start_time}")

        # if args.debug:
        #     accelerator.print(f"predictions = {predictions}")
        #     accelerator.print(f"references = {references}")

        # if args.debug:
        #     accelerator.print(f"predictions = {predictions}")
        #     accelerator.print(f"references = {references}")
        #     # ipdb.set_trace()
        # metrics.add_batch(
        #     predictions=predictions,
        #     # logits=logits,
        #     references=references,
        # )

        # flat_preds = torch.cat([flat_preds, predictions], dim=-1)
        # flat_refs = torch.cat([flat_refs, references], dim=-1)

        # flat_preds.extend(predictions)
        # flat_refs.extend(references)

        # assert flat_preds.shape == flat_refs.shape
        # accelerator.print(f"flat_preds.shape = {flat_preds.shape}")

        progress.update(1)

    # save raw results
    if accelerator.is_local_main_process:
        exp_name = args.output_dir.split("/")[-1]
        results_dirpath = os.path.join(
            utils.prm800k_validation_predictions_dirpath, exp_name, f"step_{steps}"
        )

        utils.save_pickle(raw_input_ids, os.path.join(results_dirpath, "input_ids.pkl"))
        utils.save_pickle(raw_labels, os.path.join(results_dirpath, "labels.pkl"))
        utils.save_pickle(
            all_seq_rating_logits, os.path.join(results_dirpath, "rating_logits.pkl")
        )

    # assert len(flat_preds) == len(flat_refs)
    # if args.debug:
    #     accelerator.print(f"len(flat_preds) = {len(flat_preds)}")
    #     accelerator.print(f"flat_preds[:10] = {flat_preds[:10]}")
    #     accelerator.print(f"flat_refs[:10] = {flat_refs[:10]}")

    # eval_metrics = compute_metrics(
    #     (raw_logits, raw_labels),
    #     task="prm800k" if args.prm800k else None,
    #     metrics=metrics,
    # )

    # input: raw
    # output: flat

    eval_metrics = {}

    token_rating_logits_list = []
    flat_labels = []

    for seq_logits, seq_refs in zip(all_seq_rating_logits, raw_labels):
        # left shift refs by 1
        shifted_seq_refs = seq_refs[1:]
        shifted_seq_logits = seq_logits[:-1]
        for logits, ref in zip(shifted_seq_logits, shifted_seq_refs):
            if ref != ignore_idx:
                flat_labels.append(ref)
                token_rating_logits_list.append(logits)

    # input: token_rating_logits_list, flat_labels
    # output: flat_preds, flat_refs, token_rating_probs_list

    token_rating_logits_list = torch.tensor(token_rating_logits_list)
    flat_preds = []
    flat_refs = []

    if args.prm800k:
        token_rating_probs_list = torch.softmax(token_rating_logits_list, dim=-1)
        rating_pred_idxs = token_rating_logits_list.argmax(dim=-1)  # greedy

        for pred_idx, ref in zip(rating_pred_idxs, flat_labels):
            flat_preds.append(labels[pred_idx])
            flat_refs.append(rating_token_id2label[ref])
    else:
        raise NotImplementedError("Only support PRM800K for now.")

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
            for average in MULTICLASS_AVERAGINGS:
                average_roc_auc = metric.compute(
                    references=flat_refs,
                    prediction_scores=token_rating_probs_list,
                    average=average,
                    multi_class="ovr",
                )[
                    "roc_auc"
                ]  # not metric.name!
                eval_metrics[f"roc_auc_{average}"] = average_roc_auc
            class_roc_aucs = metric.compute(
                references=flat_refs,
                prediction_scores=token_rating_probs_list,
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


def validate(model, eval_dataloader, metrics, steps):
    # pass
    logger.info("***** Running Validation *****")

    eval_metrics = inference_and_compute_metrics(model, eval_dataloader, metrics, steps)

    # accelerator.print(f"epoch {epoch}:", eval_metrics) # `accelerator.print` can print only on the main process.
    logger.info(
        f"  Step: {steps}, Validation Metrics: {eval_metrics}",
    )

    if args.with_tracking:
        metric_results = {}  # adding prefix

        for k, v in eval_metrics.items():
            metric_results["eval/" + k] = v

        # metric_results["train/global_step"] = steps
        accelerator.log(
            values=metric_results,
            step=steps,
        )

    model.train()

    if args.save_best_metric is not None:
        metric_coef = 1 if args.save_best_metric in ["loss", "perplexity"] else -1
        best_ckpt_dirpaths = list(
            pathlib.Path(args.output_dir).glob(f"step=*-best-{args.save_best_metric}=*")
        )

        assert (
            len(best_ckpt_dirpaths) <= 1
        ), "There should be at most one best checkpoint."

        if len(best_ckpt_dirpaths) == 1:
            best_ckpt_dirpath = best_ckpt_dirpaths[-1]
            best_metric = float(
                str(best_ckpt_dirpath).split("=")[-1]
            )  # step=*-best-{args.save_best_metric}=*

        else:  # len(best_ckpt_dirpath) == 0
            best_metric = None
            best_ckpt_dirpath = None

        if (
            best_metric is None
            or metric_coef * eval_metrics[args.save_best_metric]
            < metric_coef * best_metric
        ):
            if accelerator.is_local_main_process:
                if best_ckpt_dirpath is not None:
                    shutil.rmtree(best_ckpt_dirpath)
            new_best_ckpt_rel_dirpath = f"step={steps}-best-{args.save_best_metric}={eval_metrics[args.save_best_metric]}"
            save_state_to_disk(new_best_ckpt_rel_dirpath)


def train(config, args):
    # bnb config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # Load metrics
    clf_metrics = evaluate.combine(
        [evaluate.load(name) for name in utils.CLF_METRIC_NAMES]
        + [evaluate.load("roc_auc", "multiclass")]
    )

    # Load datasets
    if args.encoded_datasets_name_or_path is None:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
            )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # ::TODO::
    # if args.resume_from_checkpoint:
    #     with init_empty_weights():
    #         logger.info(
    #             "Loading model from checkpoint, so initializing model with empty weights..."
    #         )
    #         model = AutoModelForCausalLM.from_config(config)
    # else:

    if True:
        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
            )
        else:
            logger.warning("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)

    # ipdb.set_trace()
    if args.debug:
        accelerator.print(utils.nvidia_smi())

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
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
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        assert (
            num_added_tokens == 1
        ), "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})

    # add special tokens for specific tasks
    # if args.prm800k:
    #     num_added_special_tokens = tokenizer.add_special_tokens(
    #         {
    #             "cls_token": "<cls>",
    #         }
    #     )
    #     # print(tokenizer.special_tokens_map)
    #     # print(num_added_special_tokens)
    #     assert num_added_special_tokens in [
    #         0,
    #         1,
    #     ], "PRM800K should only add 1 special token - cls_token."

    # cls_token_strs = ["<pos>", "<neg>", "<neu>"]

    # cls_token_objs = [
    #     tokenizers.AddedToken(
    #         content=cls_token_str, single_word=True, lstrip=False, rstrip=False
    #     )
    #     for cls_token_str in cls_token_strs
    # ]

    # num_added_cls_tokens = tokenizer.add_tokens(cls_token_objs, special_tokens=True)

    # num_added_cls_tokens = tokenizer.add_tokens(cls_token_strs, special_tokens=True)
    # assert num_added_cls_tokens in [
    #     0,
    #     3,
    # ], "PRM800K should only add 3 special class tokens - pos_token, neg_token, neu_token."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Initialize LoRA
    if args.peft == "lora":
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Set encoding function
    if args.encoded_datasets_name_or_path is None:
        if (
            "prompt" in raw_datasets["train"].column_names
            and "completion" in raw_datasets["train"].column_names
        ):
            encode_function = partial(
                encode_with_prompt_completion_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
            )
        elif "messages" in raw_datasets["train"].column_names:
            encode_function = partial(
                encode_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
            )
        # elif args.prm800k:
        #     encode_function = partial(
        #         encode_with_problem_step_ratings_format,
        #         tokenizer=tokenizer,
        #         max_seq_length=args.max_seq_length,
        #     )
        else:
            raise ValueError(
                "You need to have either 'prompt'&'completion' or 'messages' in your column names."
            )

    if args.encoded_datasets_name_or_path is None:
        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[
                    name
                    for name in raw_datasets["train"].column_names
                    if name not in ["input_ids", "labels", "attention_mask"]
                ],
                desc="Tokenizing and reformatting SFT data",
            )
            lm_datasets.set_format(type="pt")
            lm_datasets = lm_datasets.filter(
                lambda example: (example["labels"] != -100).any()
            )
    else:
        lm_datasets = load_from_disk(args.encoded_datasets_name_or_path)

    train_dataset = lm_datasets["train"]
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.do_eval and "validation" in lm_datasets:
        validation_dataset = lm_datasets["validation"]

    # DataLoaders creation:

    if args.debug:
        padding_strategy = transformers.utils.PaddingStrategy.MAX_LENGTH
    else:
        padding_strategy = transformers.utils.PaddingStrategy.LONGEST

    # set random seed for shuffling
    if args.data_seed is not None:
        set_seed(args.data_seed)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForCausalLM(
            tokenizer=tokenizer,
            model=model,
            padding=padding_strategy,
            max_length=args.max_seq_length,
        ),
        batch_size=args.per_device_train_batch_size,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=DataCollatorForCausalLM(
            tokenizer=tokenizer,
            model=model,
            # padding=transformers.utils.PaddingStrategy.MAX_LENGTH,  # will stuck if use `LONGEST` here and not pad before gathering
            padding=padding_strategy,  # must pad before gathering
            max_length=args.max_seq_length,
        ),
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay_param_name = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_param_name)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_param_name)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    if not args.do_eval:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    else:
        (
            model,
            optimizer,
            train_dataloader,
            lr_scheduler,
            validation_dataloader,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, validation_dataloader
        )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    if args.debug:
        accelerator.print(utils.nvidia_smi())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        project_name = args.project_name
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(
            project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "resume": args.resume_tracking
                    if args.resume_tracking != ""
                    else None,
                    "id": args.resume_run_id if args.resume_run_id != "" else None,
                }
            },
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.

    completed_steps = 0
    starting_epoch = 0

    # debug
    # save_state_to_disk("test")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
        # logger.debug(args.resume_from_checkpoint)
        # accelerator.print(
        #     f"args.resume_from_checkpoint = {args.resume_from_checkpoint}"
        # )
        # logger.info(f"args.resume_from_checkpoint = {args.resume_from_checkpoint}")

        if os.path.isdir(args.resume_from_checkpoint):
            ckpt_path = args.resume_from_checkpoint
        elif args.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            ckpt_path = args.output_dir
            # dirpaths = [f.path for f in os.scandir(args.output_dir) if f.is_dir()]
            # dirpaths.sort(key=os.path.getctime)
            # ckpt_path = dirpaths[-1]  # most recent checkpoint is the last
        else:
            ckpt_path = None

        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        # load state
        accelerator.load_state(ckpt_path)
        # Extract `epoch_{i}` or `step_{i}`
        name_with_training_difference = os.path.splitext(os.path.basename(ckpt_path))[0]

        if "epoch" in name_with_training_difference:
            starting_epoch = (
                int(name_with_training_difference.replace("epoch_", "")) + 1
            )
            resume_step = None
            logger.info(f"Resuming from epoch {starting_epoch}")
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            optimizer_step = int(name_with_training_difference.replace("step_", ""))
            resume_step = optimizer_step * args.gradient_accumulation_steps
            logger.info(f"len(train_dataloader) = {len(train_dataloader)}")
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            logger.info(
                f"Resuming from epoch {starting_epoch} and step {resume_step} with gas={args.gradient_accumulation_steps}"
            )

    # update the progress_bar if load from checkpoint

    completed_steps = starting_epoch * num_update_steps_per_epoch

    if args.eval_first:
        validate(model, validation_dataloader, clf_metrics, completed_steps)

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.debug:
            accelerator.print(utils.nvidia_smi())
        if args.with_tracking:
            batch_total_loss = 0

        # # new
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # old
            # for step, batch in enumerate(train_dataloader):
            #     # We need to skip steps until we reach the resumed step
            #     if args.resume_from_checkpoint and epoch == starting_epoch:
            #         # Note 1: `completed_steps` update as the optimizer steps,
            #         # while `step` as the dataloader,
            #         # so there is a scaling factor of `gradient_accumulation_steps` between them.
            #         # Note 2: `step` starts from 0, while `completed_steps` starts from 1,
            #         # so we need to add 1 to `step` to decide whether to add 1 to `completed_steps`.
            #         # Otherwise, e.g. "step_300" , resume_step = 4800, args.gradient_accumulation_steps = 16,
            #         # completed_steps < 300 should be skipped, while completed_steps >= 300 should not;
            #         # 0 <= step < 4800 should be skipped, while step >= 4800 should not be skipped.
            #         # When step = 4784 (0 = 0 * 16, ..., 4784 = 299 * 16) => completed_steps = 299;
            #         # after `completed_steps += 1`, completed_steps = 300;
            #         # then step = 4785, completed_steps = 300, and this step batch would not be skipped,
            #         # which is not what we expect.
            #         if (
            #             resume_step is not None
            #             and completed_steps * args.gradient_accumulation_steps < resume_step
            #         ):
            #             if (
            #                 step + 1
            #             ) % args.gradient_accumulation_steps == 0:  # e.g. step = 15(0->1), 31(1->2), ..., 4783(298->299), 4799(299->300)
            #                 progress_bar.update(1)
            #                 # e.g. when step = 4799, completed_steps = 299, which is going to become 300
            #                 completed_steps += 1
            #                 # e.g. when completed_steps becomes 300, step = 4799, which is going to become 4800
            #             continue

            if args.debug:
                if accelerator.is_main_process:
                    # accelerator.print(f"batch = train_dataloader[{step}]:")
                    logger.debug(f"batch = train_dataloader[{step}]:")
                    for k, v in batch.items():
                        # accelerator.print(f"{k}: {v.shape}")
                        logger.debug(f"{k}: {v.shape}")

            # This block uses the DeepSpeed library to accelerate the training process
            with accelerator.accumulate(model):
                # Run the model on a batch of input data
                outputs = model(**batch, use_cache=False)  # not use_cache in training

                # Retrieve the loss value for the batch
                loss = outputs.loss
                # Accumulate the loss value for logging purposes
                if args.with_tracking:
                    batch_total_loss += loss.detach().float()
                # Compute gradients and update model parameters
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                # Update the learning rate scheduler
                lr_scheduler.step()
                # Flush the GPU cache if specified in the command line arguments
                if args.sync_cache_flush:
                    from deepspeed.accelerator import get_accelerator

                    get_accelerator().empty_cache()

            # checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # optimizer step
                progress_bar.update(1)
                completed_steps += 1

                # log batch/step training metrics
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(batch_total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}"
                    )

                    if args.with_tracking:
                        accelerator.log(
                            values={
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                                "train/loss": avg_loss,
                                # "train/global_step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    # restart the batch loss
                    batch_total_loss = 0

                if (
                    isinstance(checkpointing_steps, int)
                    and completed_steps % checkpointing_steps == 0
                ):
                    save_state_to_disk(f"step_{completed_steps}")

                if completed_steps % args.eval_steps == 0:
                    validate(model, validation_dataloader, clf_metrics, completed_steps)

                # break if we've reached the maximum number of training steps set
                if completed_steps >= args.max_train_steps:
                    break

        if args.epoch_checkpointing:
            save_state_to_disk(f"epoch_{epoch}")
            validate(model, validation_dataloader, clf_metrics, completed_steps)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)
        if args.peft is not None and args.peft != "":
            # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
            # and has its own save_pretrained function for only saving lora modules.
            # We have to mannually specify the is_main_process outside the save_pretrained function.
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        else:
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )


if __name__ == "__main__":
    config = {}

    # Initialize the accelerator.
    # We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs[
            "project_dir"
        ] = (
            args.output_dir
        )  # This is now `project_dir``, and you should have been seeing warnings of it being deprecated for the last few accelerate versions.

    utils.local_main_print("Initializing accelerator...")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    accelerator.print(f"accelerator.state = {accelerator.state}")

    if (
        args.gradient_accumulation_steps > 1
        and accelerator.distributed_type == DistributedType.TPU
    ):
        raise ValueError(
            "Gradient Accumulation is not yet supported for TPU."
            "Please use `--gradient_accumulation_steps 1`"
        )

    logger = get_logger(__name__)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        if args.debug:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_debug()
            evaluate.logging.set_verbosity_debug()
            evaluate.logging.enable_progress_bar()
        else:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
            evaluate.logging.set_verbosity_info()
            # evaluate.logging.set_verbosity_debug()
            evaluate.logging.enable_progress_bar()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        evaluate.logging.set_verbosity_error()
        evaluate.logging.disable_progress_bar()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    train(config, args)
