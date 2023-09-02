import argparse
import logging

import datasets
import evaluate
import torch
import transformers
import utils
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import find_executable_batch_size
from prepare_dataset import DataCollatorForCausalLM
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    OPTForCausalLM,
    SchedulerType,
    get_scheduler,
)


def compute_metrics(model, eval_dataloader, metrics):
    # set model to eval mode
    model.eval()

    progress = tqdm(
        eval_dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    )

    # samples_seen = 0 # no need since we used `Accelerator.gather_for_metrics` instead of `Accelerator.gather`:
    for step, batch in enumerate(eval_dataloader):
        accelerator.print(f"step: {step}")
        progress.update(1)
        # We can avoid the following line since we set the accelerator with `device_placement=True`.
        # batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)

        if args.prm800k:
            # rating_token_id2str = {6374: '1', 8178: '-1', 21104: '0'}
            rating_token_ids = [6374, 8178, 21104]
            rating_logits = outputs.logits[:, :, rating_token_ids]
            rating_predictions = rating_logits.argmax(dim=-1)  # greedy

            references = batch["labels"]  # shape: (batch_size, seq_len)
            ignore_idx = -100
            mask = references != ignore_idx
            predictions = rating_predictions[mask]
            references = references[mask]
        else:
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]

        predictions, references = accelerator.gather((predictions, references))

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

        metrics.add_batch(
            predictions=predictions,
            # logits=logits,
            references=references,
        )

    eval_metrics = metrics.compute()

    return eval_metrics


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test evaluation with Accelerate.",
        )

        parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=False,
        )
        parser.add_argument(
            "--tokenizer_name_or_path",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--encoded_datasets_name_or_path",
            type=str,
            default=None,
            help=("encoded datasets"),
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=8,
            help="Batch size (per device) for the training dataloader.",
        )
        parser.add_argument(
            "--max_seq_length",
            type=int,
            default=1024,
            help="The maximum total sequence length (prompt+completion) of each training example.",
        )
        parser.add_argument(
            "--prm800k",
            action="store_true",
            help=("finetune with PRM800K dataset"),
        )
        args = parser.parse_args()

        return args

    args = parse_args()

    accelerator = Accelerator()

    clf_metrics = evaluate.combine(
        [
            "accuracy",
            "f1",
            "precision",
            "recall",
            # "roc_auc"
        ]
    )

    # We now can define an inner training loop function. It should take a batch size as the only parameter,
    # and build the dataloaders in there.
    # It also gets our decorator
    @find_executable_batch_size(starting_batch_size=args.per_device_eval_batch_size)
    def inner_main(batch_size):
        # nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        # load metric

        accelerator.print(f"batch_size: {batch_size}")

        logger = get_logger(__name__)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
            evaluate.logging.set_verbosity_info()
            evaluate.logging.enable_progress_bar()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            evaluate.logging.set_verbosity_error()
            evaluate.logging.disable_progress_bar()

        accelerator.wait_for_everyone()

        tokenizer = utils.get_complete_tokenizer(args.tokenizer_name_or_path)
        assert (
            tokenizer.pad_token_id is not None
        ), "The tokenizer should have a padding token."

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            low_cpu_mem_usage=True,
        )

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        encoded_datasets = datasets.load_from_disk(args.encoded_datasets_name_or_path)

        eval_dataloader = torch.utils.data.DataLoader(
            encoded_datasets["validation"],
            shuffle=False,
            collate_fn=DataCollatorForCausalLM(
                tokenizer=tokenizer,
                model=model,
                padding="longest",
                max_length=args.max_seq_length,
            ),
            batch_size=batch_size,
        )

        # Prepare everything with `accelerator`.
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        eval_matrics = compute_metrics(model, eval_dataloader, clf_metrics)

        logger.info(f"eval_matrics: {eval_matrics}")
        accelerator.print(f"eval_matrics: {eval_matrics}")

    inner_main()
