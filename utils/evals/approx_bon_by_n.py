import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM
sys.path.append("LLaMA-Factory/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.drift import get_scores
    from utils.attribute_prompts import attribute_prompts
except ImportError:
    sys.path.append('..')
    from drift import get_scores
    from attribute_prompts import attribute_prompts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1")
parser.add_argument("--gold_cache", type=str, default="../../results/gold_scores.jsonl")
parser.add_argument("--p_path", type=str, default="../../results/preference/user1_p.json")
args = parser.parse_args()

# Load bon outputs
data_path = "../../data/bon_200.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Load gold reward cache
gold_cache = {}
with open(args.gold_cache, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold_cache[entry["prompt"]] = entry

print(f"Loaded gold reward cache for {len(gold_cache)} prompts")

# Filter BON data to only include prompts with gold scores
bon_data_filtered = [item for item in bon_data if item["prompt"] in gold_cache]
print(f"Processing {len(bon_data_filtered)} prompts (after filtering for gold cache coverage)")

if len(bon_data_filtered) == 0:
    print("ERROR: No prompts overlap between BON data and gold cache!")
    sys.exit(1)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model_path = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize tokenizer first
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Initialize model (fixed parameter order)
model = LLM(
    model=base_model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=8192,
    dtype="bfloat16",
    trust_remote_code=True,
)

# Load p vector from JSON file
with open(args.p_path, "r") as f:
    data = json.load(f)

p = None
if isinstance(data, list):
    # It's a list of entries
    for entry in data:
        if entry.get('n', entry.get('sample_size')) == 200:
            p = np.array(entry["p"])
            break
else:
    # It's a single entry
    if data.get('n', data.get('sample_size')) == 200:
        p = np.array(data["p"])

if p is None:
    raise ValueError(f"No p vector found for user {args.name} with n=200 in {args.p_path}")

print(f"Loaded p vector for user {args.name}")

# Sparsify p using torch operations for consistency
p_tensor = torch.tensor(p, device=device)
abs_p = torch.abs(p_tensor)
topk_values, topk_idx = torch.topk(abs_p, k=min(7, len(p)))  # Handle case where p has < 7 elements

p_sparse = torch.zeros_like(p_tensor)
p_sparse[topk_idx] = p_tensor[topk_idx]
p_sparse_np = p_sparse.cpu().numpy()

print(f"Sparsified p: kept top {len(topk_idx)} elements")
print(f"Non-zero elements: {torch.count_nonzero(p_sparse).item()}")

base_prompt = "You are an AI assistant."

# Choose the right attribute prompts
attribute_list = attribute_prompts  # or attribute_prompts, depending on what you want

print(f"Using {len(attribute_list)} attribute prompts")

# Precompute all scores for all outputs (up to max_k)
max_k = 20
print("Computing drift scores for all outputs...")
all_prompt_outputs = [(item["prompt"], item["outputs"]) for item in bon_data_filtered]
all_scores_full = get_scores(
    all_prompt_outputs,
    model, p_sparse_np, base_prompt, attribute_list, device, tokenizer
)

print(f"Computed scores for {len(all_scores_full)} prompts")

# Evaluate for different values of k
results_by_k = []
for k in range(2, max_k + 1, 2):
    print(f"Evaluating for k={k}...")
    selected_gold_scores = []
    all_gold_scores = []
    selected_minus_max = []
    
    for item, scores in zip(bon_data_filtered, all_scores_full):
        # Use only the first k outputs and their scores
        outputs = item["outputs"][:k]
        scores_k = scores[:k]
        
        # Find best output according to our model (fix CUDA tensor issue)
        if isinstance(scores_k, torch.Tensor):
            idx = int(torch.argmax(scores_k).cpu())  # Use torch.argmax and move to CPU
        else:
            idx = int(np.argmax(scores_k))  # Use numpy if it's already a numpy array
        
        # Get corresponding gold scores
        gold_entry = gold_cache[item["prompt"]]
        gold_score_selected = gold_entry["output_scores"][idx]
        max_gold_at_k = max(gold_entry["output_scores"][:k])
        
        selected_gold_scores.append(gold_score_selected)
        all_gold_scores.append(np.mean(gold_entry["output_scores"][:k]))
        selected_minus_max.append(max_gold_at_k - gold_score_selected)  # Fixed: max - selected
    
    # Calculate statistics
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
    
    print(f"k={k}: avg_selected_gold={avg_selected:.4f}, avg_all_gold={avg_all:.4f}, uplift={uplift:.4f}, gap_from_max={avg_selected_minus_max:.4f}")

# Save results
output_file = f"../../results/approx_bon_by_n.jsonl"
with open(output_file, "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print(f"✅ Results saved to {output_file}")

# Print summary
best_k = max(results_by_k, key=lambda x: x['uplift'])
print(f"\nBest performance at k={best_k['k']} with uplift={best_k['uplift']:.4f}")