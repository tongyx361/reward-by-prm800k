import argparse
import gzip
import json
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import blobfile as bf
import numpy as np
import orjson

Sample = Dict[str, Any]


def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)  # fallback


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def read_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


def key_by_problem(samples: List[Dict]):
    grouped_samples = defaultdict(list)
    for sample in samples:
        grouped_samples[sample["problem"]].append(sample)
    return grouped_samples


def _get_answer(sample: Sample) -> Optional[str]:
    return sample.get("answer", sample.get("given_answer", None))


def _choose_sample_by_score(samples: List[Sample], key: str) -> Optional[Sample]:
    if len(samples) == 0:
        return None
    return max(samples, key=lambda x: x[key])


__DEBUG__ = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="prm")  # one of ['orm', 'prm']
    args = parser.parse_args()
    method = args.method  # turn `args`' member into a local variable

    n_trials = 400
    ns = [10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1860]

    if __DEBUG__:
        n_trials = 1
        ns = [10]

    # samples_path = "az://openaipublic/process-supervision/scored-test-samples.jsonl"
    samples_path = (
        "/data/tongyx361/reward-by-prm800k/datasets/scored-test-samples.jsonl"
    )
    num_samples_per_problem = 1860
    all_trial_pass_rates = []

    print(f"Reading {samples_path}, this may take a while...")
    samples = read_jsonl(samples_path)
    print("Done.")
    samples_by_problem = key_by_problem(samples)  # group samples by problem
    num_problems = len(samples_by_problem)  # num of problmes

    for i in range(n_trials):
        pass_rates = []
        for n in ns:
            num_correct = 0
            for problem, problem_samples in samples_by_problem.items():
                nones = [None] * (
                    num_samples_per_problem - len(problem_samples)
                )  # ::TODO:: 为什么要混入 None？
                problem_samples = problem_samples + nones
                random.shuffle(problem_samples)
                subsamples = list(problem_samples[:n])
                if __DEBUG__:
                    print("len(subsamples)", subsamples)
                subsamples = [x for x in subsamples if x is not None]
                if __DEBUG__:
                    print("len(subsamples)", subsamples)
                subsamples = [x for x in subsamples if _get_answer(x) is not None]
                if __DEBUG__:
                    print("len(subsamples)", subsamples)
                if method == "prm":
                    choice = _choose_sample_by_score(subsamples, "prm_score")
                elif method == "orm":
                    choice = _choose_sample_by_score(subsamples, "orm_score")

                if choice is not None and choice["is_correct"]:
                    num_correct += 1
            pass_rates.append(num_correct / num_problems)
        all_trial_pass_rates.append(pass_rates)
        print(f"Trial {i}/{n_trials} {pass_rates}")

    all_trial_pass_rates = np.array(all_trial_pass_rates)
    print("Mean:", list(np.mean(all_trial_pass_rates, axis=0)))
    print("Standard deviation:", list(np.std(all_trial_pass_rates, axis=0)))


if __name__ == "__main__":
    main()
