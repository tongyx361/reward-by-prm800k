# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
from utils import (DataCollatorForCausalLM, add_pad_token,
                   get_data_module_from_encoded_datasets,
                   prm800k_compute_metrics,
                   prm800k_preprocess_logits_for_metrics)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    encoded_datasets_path: str = field(
        default=None,
        metadata={
            "help": "Path to the encoded datasets (better with no more need to preprocess)."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["eos", "pad", "unk"],
            "help": "The token used for padding. Options are 'eos', 'pad' or 'unk'.",
        },
    )
    padding_side: Optional[str] = field(
        default=None,
        # default_factory=lambda: MISSING,
        metadata={
            "choices": ["right", "left"],
            "help": 'Padding side for when sequences are padded by Trainer. Should be consistent with the previous setting. When training decoder-only models from scratch, better set padding_side to "left" for more reasonable positional encoding.',
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the `tokenizers` library) or not."
        },
    )
    use_accelerate_lib: Optional[str] = field(default="flash-attn-v2")
    eval_first: Optional[bool] = field(default=False)
    custom_debug: bool = field(default=False)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict to CPU and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank

    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        remaining_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    training_args.ddp_find_unused_parameters = not training_args.gradient_checkpointing

    local_rank = training_args.local_rank

    # Initialize config
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False  # for training
    # Set RoPE scaling factor
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=training_args.use_fast_tokenizer,
    )
    if training_args.padding_side is not None:
        tokenizer.padding_side = training_args.padding_side
    if getattr(tokenizer, "pad_token", None) is None:
        if training_args.pad_token == "unk":
            tokenizer.pad_token = tokenizer.unk_token
        elif training_args.pad_token == "eos":
            tokenizer.pad_token = tokenizer.eos_token
        elif training_args.pad_token == "pad":
            tokenizer = add_pad_token(tokenizer)

    # Load data
    data_module = get_data_module_from_encoded_datasets(data_args.encoded_datasets_path)

    # Start trainner
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForCausalLM(
            tokenizer=tokenizer,
            model=model,
            padding="max_length" if training_args.custom_debug else "longest",
            max_length=training_args.model_max_length,
        ),
        preprocess_logits_for_metrics=prm800k_preprocess_logits_for_metrics,
        compute_metrics=prm800k_compute_metrics,
        **data_module,
    )

    if training_args.eval_first:
        # trainer.evaluate()
        rank0_print("Evaluating before training...")

        class EvaluateFirstStepCallback(TrainerCallback):

            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step == 0:  # step_begin and step = 0 means before training
                    control.should_evaluate = True
        trainer.add_callback(EvaluateFirstStepCallback())

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True  # for saving
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
