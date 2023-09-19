import argparse

from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, help="model name")
args = parser.parse_args()

print("login to huggingface hub...")
login(token="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")


# model_name = "TheBloke/Llama-2-70B-fp16"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-13b-hf"
model_name = str(args.model_name)

# tokenizer_name = "hf-internal-testing/llama-tokenizer"
tokenizer_name = model_name

print(f"model_name: {model_name}")
print(f"tokenizer_name: {tokenizer_name}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, resume_download=True)

model = AutoModel.from_pretrained(
    model_name, resume_download=True, low_cpu_mem_usage=True
)
