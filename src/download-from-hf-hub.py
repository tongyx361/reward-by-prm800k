from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

login(token="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")

tokenizer_name = "hf-internal-testing/llama-tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, resume_download=True)

# model_name = "TheBloke/Llama-2-70B-fp16"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "meta-llama/Llama-2-13b-hf"

model = AutoModel.from_pretrained(
    model_name, resume_download=True, low_cpu_mem_usage=True
)
