import json
import torch
import numpy as np
import sys
from drift import drift_score_bon, get_approximation_accuracy
from attribute_prompts import attribute_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import vllm

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--p_path", type=str, required=True)
args = parser.parse_args()

# Settings
test_path = f"../data/preference/{args.name}_test.json"
p_path = args.p_path
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
base_prompt = "You are an AI assistant."

# Load test data
with open(test_path, "r") as f:
    test_data = json.load(f)

test_data = test_data[:50]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

engine = vllm.LLM(model=base_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.95, max_model_len=4096)

# Prepare data for get_approximation_accuracy
eval_data = [
    (entry["prompt"], entry["chosen"], entry["rejected"])
    for entry in test_data
]

print("Finished loading data")

# Load p vector
with open(p_path, "r") as f:
    p_list = []
    for line in f:
        entry = json.loads(line)
        if entry["user"] == args.name and entry["n"] == args.sample_size:
            p_list = np.array(entry["p"])
            break
if len(p_list) == 0:
    raise ValueError(f"No p vector found for user {args.name} in {p_path}")

abs_p = np.abs(p_list)
topk_idx = np.argsort(abs_p)[-7:]
p_sparse = np.zeros_like(p_list)
p_sparse[topk_idx] = p_list[topk_idx]

accuracy = get_approximation_accuracy(
    eval_data,
    engine,
    p_sparse,
    base_prompt,
    attribute_prompts,
    device,
    tokenizer
)

print(f"Best accuracy: {accuracy:.4f} ({int(accuracy * len(eval_data))}/{len(eval_data)})")

# Save results
with open("../results/approximation_results.jsonl", "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": args.sample_size,
        "acc": accuracy,
    }) + "\n")
print("Results saved to ../results/approximation_results.jsonl")
