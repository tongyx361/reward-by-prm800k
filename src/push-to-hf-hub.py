import json

from datasets import load_from_disk
from huggingface_hub import login
from utils import load_json

hf_token = load_json("/data/users/zhangjunlei/tyx/reward-by-prm800k/keys.json")["hf"]

login(token=hf_token)

dataset = load_from_disk("/data/users/zhangjunlei/tyx/reward-by-prm800k/datasets/prm800k-train-direct-prediction-0-02validiation-encoded-datasets")

dataset.push_to_hub("prm800k-train-direct-prediction-0-02validiation-seed42-encoded")
