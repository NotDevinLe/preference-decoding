import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
sys.path.append("LLaMA-Factory/src")
from drift import drift_score_bon_batched
from attribute_prompts import persona_prompts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
parser.add_argument("--p_path", type=str, default="../results/user_p.jsonl")
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

# Load reward model (LoRA adapter)
rm_path = f"saves/normal/user1/toy_reward_200"
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
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

# Load p vector
with open(args.p_path, "r") as f:
    p = None
    for line in f:
        entry = json.loads(line)
        if entry["user"] == args.name and entry['n'] == 200:
            p = np.array(entry["p"])
            break
if p is None:
    raise ValueError(f"No p vector found for user {args.name} in {args.p_path}")

# Sparsify p
abs_p = np.abs(p)
topk_idx = np.argpartition(abs_p, -7)[-7:]
mask = np.zeros_like(p, dtype=bool)
mask[topk_idx] = True
p_sparse = np.zeros_like(p)
p_sparse[mask] = p[mask]

base_prompt = "You are an AI assistant."

# Precompute all scores for all outputs (up to max_k)
max_k = 20
all_prompt_outputs = [(item["prompt"], item["outputs"]) for item in bon_data]
all_scores_full = drift_score_bon_batched(
    all_prompt_outputs,
    model_ds, tokenizer, base_prompt, persona_prompts, p_sparse, device, batch_size=32
)

results_by_k = []
for k in range(2, max_k + 1, 2):
    selected_gold_scores = []
    all_gold_scores = []
    selected_minus_max = []
    for item, scores in zip(bon_data, all_scores_full):
        # Use only the first k outputs and their scores
        outputs = item["outputs"][:k]
        scores_k = scores[:k]
        idx = int(np.argmax(scores_k))
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
with open("../results/approx_bon_persona_by_n.jsonl", "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print("✅ Results saved to ../results/approx_bon_persona_by_n.jsonl")