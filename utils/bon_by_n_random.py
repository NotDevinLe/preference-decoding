import os
import sys
import json
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
args = parser.parse_args()

# Load bon outputs
data_path = "../data/bon_200.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

# Load gold reward cache
gold_cache = {}
with open(args.gold_cache, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold_cache[entry["prompt"]] = entry

max_k = 20
results_by_k = []
for k in range(2, max_k + 1, 2):
    selected_gold_scores = []
    all_gold_scores = []
    selected_minus_max = []
    for item in bon_data:
        outputs = item["outputs"][:k]
        idx = random.randint(0, len(outputs) - 1)
        gold_entry = gold_cache[item["prompt"]]
        gold_score_selected = gold_entry["output_scores"][idx]
        max_gold_at_k = max(gold_entry["output_scores"][:k])
        selected_gold_scores.append(gold_score_selected)
        all_gold_scores.append(np.mean(gold_entry["output_scores"][:k]))
        selected_minus_max.append(gold_score_selected - max_gold_at_k)
    avg_selected = float(np.mean(selected_gold_scores))
    avg_all = float(np.mean(all_gold_scores))
    uplift = avg_selected - avg_all
    avg_selected_minus_max = float(np.mean(selected_minus_max))
    results_by_k.append({
        "user": args.name,
        "k": k,
        "avg_selected_gold": avg_selected,
        "avg_all_gold": avg_all,
        "uplift": uplift,
        "avg_selected_minus_max": avg_selected_minus_max
    })
    print(f"k={k}: avg_selected_gold={avg_selected:.4f}, avg_all_gold={avg_all:.4f}, uplift={uplift:.4f}, avg_selected_minus_max={avg_selected_minus_max:.4f}")

# Save results
with open("../results/approx_bon_by_n_random.jsonl", "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print("✅ Results saved to ../results/approx_bon_by_n_random.jsonl") 