import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
sys.path.append("LLaMA-Factory/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.drift import get_scores
    from utils.attribute_prompts import attribute_prompts, base_prompt
except ImportError:
    sys.path.append('..')
    from drift import get_scores
    from attribute_prompts import attribute_prompts, base_prompt
from vllm import LLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gold_cache", type=str, default="../../results/gold_scores_bon.jsonl")
parser.add_argument("--p_path", type=str, default="../../results/preference/user1_p.json")
args = parser.parse_args()

# Load bon outputs
data_path = "../../data/bon.json"
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

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

# For reward model, p is loaded from user_p.jsonl
with open(args.p_path, "r") as f:
    p = None
    p_list = json.load(f)
    for entry in p_list:
        if entry["lambda"] == 0.01:
            p = torch.tensor(entry["p"], device=device, dtype=torch.float32)  # Convert to torch tensor
            
            # Sparsify p using torch operations
            abs_p = torch.abs(p)
            topk_values, topk_idx = torch.topk(abs_p, k=7)  # Get top 7 indices
            
            # Create sparse p
            p_sparse = torch.zeros_like(p, device=device)
            p_sparse[topk_idx] = p[topk_idx]
            
            print(f"Sparsified p: kept top 7 elements (by abs value), set rest to zero.")
            print(p_sparse.cpu().numpy())  # Print as numpy for readability

            # Score all outputs for each prompt
            print("Scoring outputs with reward model...")
            all_scores = get_scores(
                [(item["prompt"], item["outputs"]) for item in bon_data],
                model, p_sparse.cpu().numpy(), base_prompt, attribute_prompts, device, tokenizer  # Convert back to numpy for get_scores
            )

            # For each prompt, select the output with the highest score
            selected_outputs = []
            gold_scores_selected = []
            gold_scores_all = []
            selected_minus_max = []
            
            for item, scores in zip(bon_data, all_scores):
                outputs = item["outputs"]
                
                # Use torch.argmax for finding best index
                idx = torch.argmax(scores).item()
                
                selected_outputs.append({
                    "prompt": item["prompt"],
                    "output": outputs[idx],
                    "score": scores[idx].item()  # Convert to Python scalar
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
                
            print(f"Selected best outputs for {len(selected_outputs)} prompts.")

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
            with open("../../results/approx_bon.jsonl", "a") as f:
                f.write(json.dumps({
                    "user": args.name,
                    "n": entry["sample_size"],
                    "lambda": entry["lambda"],
                    "uplift": uplift,
                    "selected_minus_max": avg_selected_minus_max
                }) + "\n")
            print(f"✅ Results saved to ../../results/approx_bon.jsonl")