import json
import torch
import numpy as np
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import vllm
import os

sys.path.append("..")
from drift import get_approximation_accuracy
from attribute_prompts import attribute_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
# parser.add_argument("--num_expectation_samples", type=int, required=True)
# parser.add_argument("--p_path", type=str, required=True)
parser.add_argument("--k", type=int, default=7)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

# Settings
test_path = f"../../data/preference/{args.name}_test.json"
# p_path = args.p_path
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
base_prompt = "You are an AI assistant."

# Load test data
with open(test_path, "r") as f:
    test_data = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

engine = vllm.LLM(model=base_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)
# model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)

# Prepare data for get_approximation_accuracy
eval_data = [
    (entry["prompt"], entry["chosen"], entry["rejected"])
    for entry in test_data
]

print("Finished loading data")

def sparsify_p(p_list, k=14):
    p_list = np.array(p_list)
    abs_p = np.abs(p_list)
    topk_idx = np.argsort(abs_p)[-k:]
    p_sparse = np.zeros_like(p_list)
    p_sparse[topk_idx] = p_list[topk_idx]
    return p_sparse

eval_data = eval_data[:args.sample_size]

p_path = f"../../data/preference/user1_p_1500.jsonl"

selected_attr_idx = [
    1,
    2,
    3,
    6,
    9,
    10,
    20,
    30,
    31,
    32,
    36,
    37,
    57
  ]
attribute_prompts = [attribute_prompts[i] for i in selected_attr_idx]

with open(p_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        p = entry["p"]
        p = np.array(p)
        p = p / np.linalg.norm(p)
        p = p.tolist()

        accuracy = get_approximation_accuracy(
            eval_data,
            engine,
            p,
            base_prompt,
            attribute_prompts,
            device,
            tokenizer
        )

        print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(eval_data))}/{len(eval_data)})")

        # Save results
        with open(args.save_path, "a") as f:
            f.write(json.dumps({
                "user": args.name,
                "n": args.sample_size,
                "acc": accuracy,
                "k": args.k,
            }) + "\n")
        print(f"Results saved to {args.save_path}")