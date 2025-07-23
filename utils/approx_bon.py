import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
sys.path.append("LLaMA-Factory/src")
from drift import drift_score_bon_batched
from attribute_prompts import attribute_prompts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
args = parser.parse_args()

# Load bon outputs
data_path = "../data/bon_200.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

bon_data = bon_data[:100]
print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Load gold reward cache
gold_cache = {}
with open(args.gold_cache, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold_cache[entry["prompt"]] = entry
print(f"Loaded gold reward cache for {len(gold_cache)} prompts from {args.gold_cache}")

# Load reward model (LoRA adapter)
rm_path = f"saves/normal/user1/toy_reward_{args.sample_size}"
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading reward model from: {rm_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_ds = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# For reward model, p is loaded from user_p.jsonl
with open("../results/user_p.jsonl", "r") as f:
    p = None
    for line in f:
        entry = json.loads(line)
        if entry["user"] == args.name and entry['n'] == args.sample_size:
            p = np.array(entry["p"])
            break
if p is None:
    raise ValueError(f"No p vector found for user {args.name} in ../results/user_p.jsonl")

# Sparsify p
abs_p = np.abs(p)
topk_idx = np.argpartition(abs_p, -7)[-7:]
mask = np.zeros_like(p, dtype=bool)
mask[topk_idx] = True
p_sparse = np.zeros_like(p)
p_sparse[mask] = p[mask]
print(f"Sparsified p: kept top 7 elements (by abs value), set rest to zero.")
print(p_sparse)

base_prompt = "You are an AI assistant."

# Score all outputs for each prompt
print("Scoring outputs with reward model...")
all_scores = drift_score_bon_batched(
    [(item["prompt"], item["outputs"]) for item in bon_data],
    model_ds, tokenizer, base_prompt, attribute_prompts, p_sparse, device, batch_size=8
)

# For each prompt, select the output with the highest score
selected_outputs = []
gold_scores_selected = []
gold_scores_all = []
selected_minus_max = []
for item, scores in zip(bon_data, all_scores):
    outputs = item["outputs"]
    idx = int(np.argmax(scores))
    selected_outputs.append({
        "prompt": item["prompt"],
        "output": outputs[idx],
        "score": scores[idx]
    })
    # Use gold cache for this prompt
    gold_entry = gold_cache[item["prompt"]]
    gold_score_selected = gold_entry["output_scores"][idx]
    gold_scores_selected.append(gold_score_selected)
    gold_scores_all.append(np.mean(gold_entry["output_scores"]))
    max_gold = max(gold_entry["output_scores"])
    selected_minus_max.append(max_gold - gold_score_selected)
print(f"Selected best outputs for {len(selected_outputs)} prompts.")

avg_gold_reward_selected = float(np.mean(gold_scores_selected))
avg_gold_reward_all = float(np.mean(gold_scores_all))
uplift = avg_gold_reward_selected - avg_gold_reward_all
avg_selected_minus_max = float(np.mean(selected_minus_max))
print(f"Average gold reward (selected): {avg_gold_reward_selected:.4f}")
print(f"Average gold reward (all): {avg_gold_reward_all:.4f}")
print(f"Uplift (selected - all): {uplift:.4f}")
print(f"max gold RM - Average (selected): {avg_selected_minus_max:.4f}")

# Save results in the required format
with open("../results/approx_bon.jsonl", "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": args.sample_size,
        "uplift": uplift,
        "selected_minus_max": avg_selected_minus_max
    }) + "\n")
print(f"✅ Results saved to ../results/approx_bon.jsonl")
