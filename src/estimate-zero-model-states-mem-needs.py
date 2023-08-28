from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)
from transformers import AutoModel

model_name = "meta-llama/Llama-2-13b-hf"
model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)

zero_stage2api = {
    2: estimate_zero2_model_states_mem_needs_all_live,
    3: estimate_zero3_model_states_mem_needs_all_live,
}

for zero_stage in [3, 2]:
    print(f"# zero_stage = {zero_stage}")
    for num_gpus_per_node in [8, 6, 4, 2]:
        print(f"## num_gpus_per_node = {num_gpus_per_node}")
        zero_stage2api[zero_stage](
            model, num_gpus_per_node=num_gpus_per_node, num_nodes=1
        )
