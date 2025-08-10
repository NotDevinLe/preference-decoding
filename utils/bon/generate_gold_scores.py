import os
import sys
import json
import torch
import numpy as np
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
parser = argparse.ArgumentParser(description="Generate gold scores for BON dataset")
parser.add_argument("--gold_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                    help="Gold model to use for scoring")
parser.add_argument("--p_path", type=str, default="../../results/preference/user1_p.json",
                    help="Path to p vector file")
parser.add_argument("--output_path", type=str, default="../../results/gold_scores_bon_200.jsonl",
                    help="Path to save gold scores")
parser.add_argument("--max_outputs", type=int, default=20,
                    help="Maximum number of outputs to score per prompt (default: 20)")
parser.add_argument("--lambda_value", type=float, default=1.0,
                    help="Lambda value to use for gold p vector (default: 1.0)")
args = parser.parse_args()

# Load bon outputs
data_path = "../../data/bon.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

print(f"Loaded {len(bon_data)} prompts from {data_path}")
print(f"Each prompt has {len(bon_data[0]['outputs'])} outputs")
print(f"Will score first {args.max_outputs} outputs per prompt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"\nLoading gold model: {args.gold_model}")
gold_model = LLM(model=args.gold_model, tensor_parallel_size=1, 
                 gpu_memory_utilization=0.8, max_model_len=8192)

gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_model)
gold_tokenizer.pad_token = gold_tokenizer.eos_token

# Load gold p vector
with open(args.p_path, "r") as f:
    p_list = json.load(f)
    gold_p = None
    
    # First try to find the requested lambda
    for entry in p_list:
        if abs(entry["lambda"] - args.lambda_value) < 1e-6:
            gold_p = torch.tensor(entry["p"], device=device, dtype=torch.float32)
            print(f"Found p vector for lambda={args.lambda_value}")
            break
    
    # If not found, use fallback
    if gold_p is None:
        print(f"Lambda={args.lambda_value} not found, trying lambda=0.01 as fallback")
        for entry in p_list:
            if abs(entry["lambda"] - 0.01) < 1e-6:
                gold_p = torch.tensor(entry["p"], device=device, dtype=torch.float32)
                print(f"Using p vector for lambda=0.01")
                break
    
    if gold_p is None:
        raise ValueError("Could not find suitable p vector in file")

print(f"Gold p vector shape: {gold_p.shape}")
print(f"Gold p vector (first 10 elements): {gold_p[:10].cpu().numpy()}")

# Generate gold scores
print("\nGenerating gold scores...")
gold_cache = {}

for i, item in enumerate(bon_data):
    prompt = item["prompt"]
    # Score first max_outputs outputs
    outputs_to_score = item["outputs"][:args.max_outputs]
    
    print(f"[{i+1}/{len(bon_data)}] Scoring {len(outputs_to_score)} outputs for prompt: {prompt[:50]}...")
    
    # Get gold scores
    scores = get_scores(
        [(prompt, outputs_to_score)],
        gold_model, gold_p.cpu().numpy(), base_prompt, attribute_prompts, device, gold_tokenizer
    )[0]  # Get first (and only) result
    
    # Convert scores to list
    if hasattr(scores, 'tolist'):
        scores_list = scores.tolist()
    else:
        scores_list = list(scores)
    
    gold_cache[prompt] = {
        "prompt": prompt,
        "output_scores": scores_list,
        "num_outputs_scored": len(outputs_to_score),
        "best_idx": int(np.argmax(scores_list)),
        "best_score": float(max(scores_list)),
        "avg_score": float(np.mean(scores_list)),
        "std_score": float(np.std(scores_list))
    }
    
    # Print some statistics
    if (i + 1) % 10 == 0:
        print(f"  Best score: {gold_cache[prompt]['best_score']:.4f}")
        print(f"  Avg score: {gold_cache[prompt]['avg_score']:.4f}")
        print(f"  Std score: {gold_cache[prompt]['std_score']:.4f}")

# Save gold cache
print(f"\nSaving gold cache to {args.output_path}")
with open(args.output_path, "w") as f:
    for prompt, entry in gold_cache.items():
        f.write(json.dumps(entry) + "\n")

# Print summary statistics
all_best_scores = [entry["best_score"] for entry in gold_cache.values()]
all_avg_scores = [entry["avg_score"] for entry in gold_cache.values()]

print("\n=== SUMMARY ===")
print(f"✅ Gold scores generated for {len(gold_cache)} prompts")
print(f"Outputs scored per prompt: {args.max_outputs}")
print(f"Overall best score mean: {np.mean(all_best_scores):.4f}")
print(f"Overall average score mean: {np.mean(all_avg_scores):.4f}")
print(f"Score improvement potential: {np.mean(all_best_scores) - np.mean(all_avg_scores):.4f}")
print(f"Results saved to: {args.output_path}")