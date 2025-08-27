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

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n_values", type=str, default="5,10,15,20,50,100,150,200",
                        help="Comma-separated list of n values for best-of-n sampling")
    parser.add_argument("--training_size", type=int, default=200,
                        help="Number of training data points")
    parser.add_argument("--lambda_val", type=float, default=0.01,
                        help="Lambda parameter value")
    parser.add_argument("--output_path", type=str, default="../../results/drift_bon_indices.jsonl",
                        help="Path to save indices and lambda")
    args = parser.parse_args()

    # Load bon outputs
    data_path = "../../data/bon_attributes.json"
    with open(data_path, "r") as f:
        bon_data = json.load(f)
    
    p_path = f"../../results/user1_p.jsonl"

    print(f"Loaded {len(bon_data)} prompts from {data_path}")
    print(f"Each prompt has {len(bon_data[0]['outputs'])} outputs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model and tokenizer setup for reward model
    print("\n=== SETTING UP REWARD MODEL ===")
    small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading reward model: {small_model_id}")
    model = LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)

    tokenizer = AutoTokenizer.from_pretrained(small_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # selected_indices = [0, 1, 2, 31, 33, 37, 43]
    # attribute_prompts = [attribute_prompts[i] for i in selected_indices]

    # Load p vector for reward model from JSONL file
    p_sparse = None
    with open(p_path, "r") as f:
        for line in f:
            p_entry = json.loads(line.strip())

            if p_entry["lambda0"] != args.lambda_val:
                continue
            
            p = torch.tensor(p_entry["p"], device=device, dtype=torch.float32)
            
            # Sparsify p using torch operations
            abs_p = torch.abs(p)
            topk_values, topk_idx = torch.topk(abs_p, k=7)
            
            # Create sparse p
            p_sparse = torch.zeros_like(p, device=device)
            p_sparse[topk_idx] = p[topk_idx]
            
            # Parse n values for best-of-n sampling
            n_values = [int(x) for x in args.n_values.split(",")]
            print(f"\n=== EVALUATING BEST-OF-N FOR N VALUES: {n_values} ===")

            results = []

            for n in n_values:
                print(f"\n--- Evaluating with n={n} outputs ---")
                
                # For each prompt, use only first n outputs
                bon_data_subset = []
                for item in bon_data:
                    bon_data_subset.append({
                        "prompt": item["prompt"],
                        "outputs": item["outputs"][:n]  # Use only first n outputs
                    })
                
                # Score all outputs with reward model
                all_scores = get_scores(
                    [(item["prompt"], item["outputs"]) for item in bon_data_subset],
                    model, p_sparse.cpu().numpy(), base_prompt, attribute_prompts, device, tokenizer
                )
                
                # For each prompt, get the index of the output with the highest reward model score
                selected_indices = []
                
                for item, scores in zip(bon_data_subset, all_scores):
                    # Find best according to reward model
                    reward_idx = torch.argmax(scores).item()
                    selected_indices.append(reward_idx)
                
                print(f"Results for n={n}:")
                print(f"  Number of prompts: {len(selected_indices)}")
                print(f"  Selected indices: {selected_indices[:10]}...")  # Show first 10
                
                result = {
                    "user": args.name,
                    "n": n,
                    "training_size": args.training_size,
                    "lambda": p_entry["lambda0"],
                    "selected_indices": selected_indices,
                    "num_prompts": len(selected_indices)
                }
                results.append(result)
                
                # Save to JSONL file
                with open(args.output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
            break

    print(f"\n✅ Results saved to {args.output_path}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'n':<10} {'Num Prompts':<15} {'Lambda':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['n']:<10} {r['num_prompts']:<15} {r['lambda']:<10}")