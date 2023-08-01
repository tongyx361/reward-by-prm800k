from transformers import AutoModel
from huggingface_hub import login

login(token="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")

model_name = "TheBloke/Llama-2-70B-fp16"
model = AutoModel.from_pretrained(model_name, resume_download=True)