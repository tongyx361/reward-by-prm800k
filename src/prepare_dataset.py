from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


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
        max_feature_length = max(len(label) for label in labels)

        # truncate
        if self.max_length is not None:
            for feature in features:
                for k, v in feature.items():
                    if len(v) > self.max_length:
                        feature[k] = v[: self.max_length]
            if self.padding == "max_length" or self.padding == PaddingStrategy.MAX_LENGTH:
                max_feature_length = self.max_length

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            if self.pad_to_multiple_of is not None:
                max_feature_length = (
                    (max_feature_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_feature_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:  # padding_side == "left"
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

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
