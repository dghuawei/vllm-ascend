import os
import json
import argparse
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True, help="path to config.json of the base model")
parser.add_argument("--save-path", required=True, help="directory where the lora is saved")
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--alpha", type=int, default=32)
args = parser.parse_args()

f = open(os.path.join(args.model_path, "config.json"))
config = json.load(f)
f.close()

hidden_size = config["hidden_size"]
moe_intermediate_size = config["moe_intermediate_size"]
n_experts = config["n_routed_experts"]
n_shared = config["n_shared_experts"]
shared_intermediate_size = moe_intermediate_size * n_shared
n_layers = config["num_hidden_layers"]
rank = args.rank

print("hidden_size =", hidden_size)
print("moe_intermediate_size =", moe_intermediate_size)
print("n_routed_experts =", n_experts)
print("n_shared_experts =", n_shared)
print("num_hidden_layers =", n_layers)

state_dict = {}

for layer in range(n_layers):
    for expert in range(n_experts):
        prefix = "base_model.model.model.layers." + str(layer) + ".mlp.experts." + str(expert)

        # gate_proj: weight is [moe_intermediate_size, hidden_size]
        state_dict[prefix + ".gate_proj.lora_A.weight"] = torch.zeros(rank, hidden_size, dtype=torch.bfloat16)
        state_dict[prefix + ".gate_proj.lora_B.weight"] = torch.zeros(moe_intermediate_size, rank, dtype=torch.bfloat16)

        # up_proj: weight is [moe_intermediate_size, hidden_size]
        state_dict[prefix + ".up_proj.lora_A.weight"] = torch.zeros(rank, hidden_size, dtype=torch.bfloat16)
        state_dict[prefix + ".up_proj.lora_B.weight"] = torch.zeros(moe_intermediate_size, rank, dtype=torch.bfloat16)

        # down_proj: weight is [hidden_size, moe_intermediate_size]
        state_dict[prefix + ".down_proj.lora_A.weight"] = torch.zeros(rank, moe_intermediate_size, dtype=torch.bfloat16)
        state_dict[prefix + ".down_proj.lora_B.weight"] = torch.zeros(hidden_size, rank, dtype=torch.bfloat16)

    # shared expert, it is one module per layer, without a number
    prefix = "base_model.model.model.layers." + str(layer) + ".mlp.shared_experts"

    state_dict[prefix + ".gate_proj.lora_A.weight"] = torch.zeros(rank, hidden_size, dtype=torch.bfloat16)
    state_dict[prefix + ".gate_proj.lora_B.weight"] = torch.zeros(shared_intermediate_size, rank, dtype=torch.bfloat16)

    state_dict[prefix + ".up_proj.lora_A.weight"] = torch.zeros(rank, hidden_size, dtype=torch.bfloat16)
    state_dict[prefix + ".up_proj.lora_B.weight"] = torch.zeros(shared_intermediate_size, rank, dtype=torch.bfloat16)

    state_dict[prefix + ".down_proj.lora_A.weight"] = torch.zeros(rank, shared_intermediate_size, dtype=torch.bfloat16)
    state_dict[prefix + ".down_proj.lora_B.weight"] = torch.zeros(hidden_size, rank, dtype=torch.bfloat16)

os.makedirs(args.save_path, exist_ok=True)
save_file(state_dict, os.path.join(args.save_path, "adapter_model.safetensors"))

adapter_config = {
    "base_model_name_or_path": args.model_path,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "lora_alpha": args.alpha,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": rank,
    "target_modules": ["down_proj", "gate_proj", "up_proj"],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}

f = open(os.path.join(args.save_path, "adapter_config.json"), "w")
json.dump(adapter_config, f, indent=2)
f.close()

print("saved", len(state_dict), "tensors to", args.save_path)
