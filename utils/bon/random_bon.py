
import json
import torch
import numpy as np
import random
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
args = parser.parse_args()

# Load bon outputs
data_path = "../data/bon.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Load gold reward cache
gold_cache = {}
with open(args.gold_cache, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold_cache[entry["prompt"]] = entry
print(f"Loaded gold reward cache for {len(gold_cache)} prompts from {args.gold_cache}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
random.seed(42)

print("Randomly selecting outputs...")

# For each prompt, randomly select an output
selected_outputs = []
gold_scores_selected = []
gold_scores_all = []
selected_minus_max = []

print(len(bon_data))
for item in bon_data:
    outputs = item["outputs"]
    
    # Randomly select an output
    idx = random.randint(0, len(outputs) - 1)
    
    selected_outputs.append({
        "prompt": item["prompt"],
        "output": outputs[idx],
        "score": 0.0  # No score for random selection
    })
    
    # Use gold cache for this prompt
    gold_entry = gold_cache[item["prompt"]]
    gold_score_selected = gold_entry["output_scores"][idx]
    gold_scores_selected.append(gold_score_selected)
    gold_scores_all.append(float(torch.tensor(gold_entry["output_scores"]).mean().item()))
    
    # Convert to torch tensor for max operation
    gold_scores_tensor = torch.tensor(gold_entry["output_scores"])
    max_gold = torch.max(gold_scores_tensor).item()
    selected_minus_max.append(max_gold - gold_score_selected)
    
print(f"Randomly selected outputs for {len(selected_outputs)} prompts.")

# Convert lists to torch tensors for statistics
gold_scores_selected_tensor = torch.tensor(gold_scores_selected, dtype=torch.float32)
gold_scores_all_tensor = torch.tensor(gold_scores_all, dtype=torch.float32)
selected_minus_max_tensor = torch.tensor(selected_minus_max, dtype=torch.float32)

# Calculate statistics using torch
avg_gold_reward_selected = gold_scores_selected_tensor.mean().item()
avg_gold_reward_all = gold_scores_all_tensor.mean().item()
uplift = avg_gold_reward_selected - avg_gold_reward_all
avg_selected_minus_max = selected_minus_max_tensor.mean().item()

print(f"Average gold reward (selected): {avg_gold_reward_selected:.4f}")
print(f"Average gold reward (all): {avg_gold_reward_all:.4f}")
print(f"Uplift (selected - all): {uplift:.4f}")
print(f"max gold RM - Average (selected): {avg_selected_minus_max:.4f}")

# Save results in the required format
with open("../results/random_bon.jsonl", "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": len(bon_data),  # Use number of prompts as n
        "uplift": uplift,
        "selected_minus_max": avg_selected_minus_max
    }) + "\n")
print(f"✅ Results saved to ../results/random_bon.jsonl")