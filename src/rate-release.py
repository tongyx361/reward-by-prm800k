# %%
from vllm import LLM, SamplingParams
import os
import gzip
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import random
from transformers import AutoTokenizer
from functools import reduce
import pickle
import time

import blobfile as bf
import numpy as np
import orjson


# %%
# model_name_or_path = "/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
model_name_or_path = "/data/users/zhangjunlei/tyx/reward-by-prm800k/models/direct-prediction/meta-llama/Llama-2-7b-hf"
tokenizer_name_or_path = "/data/users/zhangjunlei/tyx/.cache/huggingface/hub/models--hf-internal-testing--llama-tokenizer/snapshots/99eceeba6e8289bee767f0771166b5917e70e470"
gpu_ids = [1,3,5,7]
# gpu_ids = [0,6]
# gpu_ids = [0]
tensor_parallel_size=len(gpu_ids)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids])

samples_to_rate_path = "/data/users/zhangjunlei/tyx/reward-by-prm800k/datasets/prm800k-scored-test-samples.jsonl"
input_ids_path = "/data/users/zhangjunlei/tyx/reward-by-prm800k/datasets/gpt4-generated-math-solutions-till-each-step-input-ids-list.pkl"

rating2prob_list_path = f"/data/users/zhangjunlei/tyx/reward-by-prm800k/datasets/gpt4-generated-math-solutions-till-each-step-rating2prob-list-{'-'.join([str(i) for i in gpu_ids])}.pkl"
rated_samples_path = f"/data/users/zhangjunlei/tyx/reward-by-prm800k/eval/rated-samples/gpt-4-generatations/llama-2-7b-2023-08-15-2-step-2040-ratings--{'-'.join([str(i) for i in gpu_ids])}.jsonl"

__DEBUG__ = False
# __DEBUG__ = True
__DEBUG_FOR__ = {
    # "inference_sample_num": 100
    "inference_sample_num": None
}

# %%
def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)  # fallback


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def _read_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


# %%
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# rating2word = {1: "positive", -1: "negative", 0: "neutral"}
# rating_words = list(rating2word.values())
# rating_token_ids = tokenizer(rating_words, add_special_tokens=False).input_ids # [[6374], [8178], [21104]]
# rating_token_ids = set([token_id[0] for token_id in rating_token_ids]) # [6374, 8178, 21104]
# # print(rating_token_ids)
# token_id2rating_str = {tokenizer(word, add_special_tokens=False).input_ids[0]: str(rating) for rating, word in rating2word.items()}
# # print(token_id2rating_str)
rating_token_ids = [6374, 8178, 21104]
token_id2rating_str = {6374: '1', 8178: '-1', 21104: '0'}

# %%
prompt_input_ids_list = None

print("Loading input_ids_list...")
strat_time = time.time()

f = open(input_ids_path, "rb")
prompt_input_ids_list = pickle.load(f)
f.close()
print("input_ids_list loaded! Time used: ", time.time() - strat_time)

# %%
prompts = None

# print("Loading generated_samples...")
# strat_time = time.time()
# generated_samples = _read_jsonl(samples_to_rate_path)
# print("generated_samples loaded! Time used: ", time.time() - strat_time)

# print("Extracting prompts...")
# strat_time = time.time()
# prompts = []
# for sample in generated_samples:
#     problem = sample["problem"]
#     steps = sample["steps"]
#     solution_so_far = problem
#     solution_until_step_idx = []
#     for step in steps:
#         solution_so_far += "\n" + step
#         solution_until_step_idx.append(solution_so_far)
#     prompts += solution_until_step_idx
#     if __DEBUG__: break
# print("prompts extracted! Time used: ", time.time() - strat_time)

# %%
print("Loading llm...")
strat_time = time.time()
llm = LLM(
    model=model_name_or_path,
    tokenizer=tokenizer_name_or_path,
    tokenizer_mode="auto",
    trust_remote_code=True,
    tensor_parallel_size=tensor_parallel_size,
    dtype="auto",
    seed=0,
)
print("llm loaded! Time used: ", time.time() - strat_time)

# %%
top_k = 3
sampling_params = SamplingParams(
    temperature=1,
    top_p=1,
    top_k=top_k,
    max_tokens=1,
    logprobs=top_k,
)

# new_sample = sample.copy()

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("Start generating...")
strat_time = time.time()
if __DEBUG_FOR__["inference_sample_num"] is not None:
    outputs = llm.generate(prompts=prompts[:__DEBUG_FOR__["inference_sample_num"]], prompt_token_ids=prompt_input_ids_list[:__DEBUG_FOR__["inference_sample_num"]], sampling_params=sampling_params)
else:
    outputs = llm.generate(prompts=prompts, prompt_token_ids=prompt_input_ids_list, sampling_params=sampling_params)
print("Generating completed! Time used: ", time.time() - strat_time)

# %%
rating2prob_list = []
for idx, output in enumerate(outputs):
    token_id2logprob = output.outputs[0].logprobs[0]

    top_token_ids = token_id2logprob.keys()

    logprobs = list(token_id2logprob.values())
    probs = np.exp(logprobs)
    sum_probs = sum(probs)

    top_p = 0.95
    if sum_probs < top_p:
        print(f"{idx}: sum_probs < {top_p}",)

    norm_probs = probs / sum_probs

    if set(top_token_ids) != set(rating_token_ids):
        print(f"{idx}: set(top_token_ids) != set(rating_token_ids):")

    rating_strs = [token_id2rating_str.get(top_token_id, top_token_id) for top_token_id in top_token_ids ]
    rating_probs = {
        rating_str: norm_prob for rating_str, norm_prob in zip(rating_strs, norm_probs)
    }
    rating2prob_list.append(rating_probs)

print("Saving rating2prob_list...")
strat_time = time.time()
f = open(rating2prob_list_path, "wb")
pickle.dump(rating2prob_list, f)
f.close()
print("rating2prob_list saved!  Time used: ", time.time() - strat_time)

# %%
total_step_num_so_far = 0
for sample in generated_samples:
    sample_step_num_so_far = 0
    for step_idx, step in enumerate(sample["steps"]):
        rating2prob = rating2prob_list[total_step_num_so_far]

        rating2prob = {rating: float(prob) for rating, prob in rating2prob.items()}

        sample["rating_probs"][step_idx] = rating2prob

        total_step_num_so_far += 1

    sample["orm_score"] = None

    positive_probs = [rating2prob["1"] for rating2prob in sample["rating_probs"]]
    sample["prm_score"] = {
        "positive_probs_product": float(np.prod(positive_probs)),
        "positive_probs_minimum": float(np.min(positive_probs)),
    }


# %%

print("Saving rated_samples...")
strat_time = time.time()
# 打开文件以写入JSONL格式
with open(rated_samples_path, 'w') as f:
    # 迭代列表中的每个元素
    for item in generated_samples:
        # 将列表中的每个元素转换为JSON字符串
        json_str = orjson.dumps(item).decode()

        # 将JSON字符串写入文件，并添加换行符以分隔每个元素
        f.write(json_str + '\n')
f.close()
print("rated_samples saved! Time used: ", time.time() - strat_time)
