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

# Import LLaMA-Factory modules for loading trained models
from llamafactory.hparams import ModelArguments, FinetuningArguments
from llamafactory.model import load_tokenizer, load_model

def format_llama3_prompt(prompt: str, response: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
    )

def get_score(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        _, _, values = model(**inputs)
        return values[:, -1]

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--gold_cache", type=str, default="../../results/gold_scores_bon.jsonl")
    parser.add_argument("--p_path", type=str, default="../../results/preference/user1_p.json")
    parser.add_argument("--gold_model_path", type=str, required=True,
                        help="Path to the trained gold reward model adapter")
    parser.add_argument("--generate_gold", action="store_true", 
                        help="Generate gold scores for first 20 outputs")
    parser.add_argument("--gold_only", action="store_true",
                        help="Only generate gold scores, skip reward model evaluation")
    parser.add_argument("--n_values", type=str, default="5,10,15,20,50,100,150,200",
                        help="Comma-separated list of n values for best-of-n sampling")
    parser.add_argument("--training_size", type=int, default=200,
                        help="Number of training data points")
    args = parser.parse_args()

    # Load bon outputs
    data_path = "../../data/bon.json"
    with open(data_path, "r") as f:
        bon_data = json.load(f)

    print(f"Loaded {len(bon_data)} prompts from {data_path}")
    print(f"Each prompt has {len(bon_data[0]['outputs'])} outputs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only load gold model if we need to generate gold scores
    gold_model = None
    gold_tokenizer = None

    # Generate gold scores if requested
    if args.generate_gold:
        print("\n=== GENERATING GOLD SCORES ===")
        
        # Load gold reward model
        print(f"Loading gold reward model from: {args.gold_model_path}")
        gold_model_args = ModelArguments(
            model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
            adapter_name_or_path=args.gold_model_path,
            trust_remote_code=True,
            use_fast_tokenizer=True,
        )
        gold_finetuning_args = FinetuningArguments(stage="rm")
        
        gold_tokenizer_module = load_tokenizer(gold_model_args)
        gold_tokenizer = gold_tokenizer_module["tokenizer"]
        if gold_tokenizer.pad_token is None:
            gold_tokenizer.pad_token = gold_tokenizer.eos_token
        
        gold_model = load_model(
            tokenizer=gold_tokenizer,
            model_args=gold_model_args,
            finetuning_args=gold_finetuning_args,
            is_trainable=False,
            add_valuehead=True
        )
        gold_model.to(device)
        gold_model.eval()
        
        # Generate gold scores for first 20 outputs of each prompt
        print("Generating gold scores for first 20 outputs per prompt using trained gold model...")
        gold_cache = {}
        
        for i, item in enumerate(bon_data):
            prompt = item["prompt"]
            # Only score first 20 outputs to save time
            outputs_to_score = item["outputs"][:20]
            
            # Get gold scores using the trained reward model directly
            scores = []
            for output in outputs_to_score:
                formatted_text = format_llama3_prompt(prompt, output)
                score = get_score(gold_model, gold_tokenizer, formatted_text)
                scores.append(score.item())
            
            gold_cache[prompt] = {
                "prompt": prompt,
                "output_scores": scores,
                "num_outputs_scored": len(outputs_to_score)
            }
            
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(bon_data)}] Scored {len(outputs_to_score)} outputs for prompt: {prompt[:50]}...")
        
        # Save gold cache
        print(f"\nSaving gold cache to {args.gold_cache}")
        with open(args.gold_cache, "w") as f:
            for prompt, entry in gold_cache.items():
                f.write(json.dumps(entry) + "\n")
        
        print(f"✅ Gold scores generated and saved for {len(gold_cache)} prompts")
        
        # Clean up gold model completely
        del gold_model
        del gold_tokenizer
        del gold_model_args
        del gold_finetuning_args
        del gold_tokenizer_module
        torch.cuda.empty_cache()
        
        print("Gold model unloaded from memory")
        
        # Exit if only generating gold scores
        if args.gold_only:
            print("\n✅ Gold score generation complete. Exiting (--gold_only flag set).")
            sys.exit(0)

    # Load gold reward cache
    print("\n=== LOADING GOLD CACHE ===")
    gold_cache = {}
    with open(args.gold_cache, "r") as f:
        for line in f:
            entry = json.loads(line)
            gold_cache[entry["prompt"]] = entry
    print(f"Loaded gold reward cache for {len(gold_cache)} prompts from {args.gold_cache}")

    # Model and tokenizer setup for reward model
    print("\n=== SETTING UP REWARD MODEL ===")
    small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading reward model: {small_model_id}")
    model = LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)

    tokenizer = AutoTokenizer.from_pretrained(small_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load p vector for reward model
    with open(args.p_path, "r") as f:
        p_list = json.load(f)
        for p in p_list:
            if p["sample_size"] == args.training_size:
                p = torch.tensor(p["p"], device=device, dtype=torch.float32)
                
                # Sparsify p using torch operations
                abs_p = torch.abs(p)
                topk_values, topk_idx = torch.topk(abs_p, k=7)
                
                # Create sparse p
                p_sparse = torch.zeros_like(p, device=device)
                p_sparse[topk_idx] = p[topk_idx]
                
                print(f"Sparsified p: kept top 7 elements (by abs value), set rest to zero.")
                print(p_sparse.cpu().numpy())
                break

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
        
        # For each prompt, select the output with the highest reward model score
        gold_scores_selected = []
        
        for item, scores in zip(bon_data_subset, all_scores):
            outputs = item["outputs"]
            
            # Find best according to reward model
            reward_idx = torch.argmax(scores).item()
            
            # Get gold scores for this prompt (only first n)
            gold_entry = gold_cache[item["prompt"]]
            gold_scores_subset = gold_entry["output_scores"][:n]
            
            # Gold score of reward-selected output
            gold_score_selected = gold_scores_subset[reward_idx]
            gold_scores_selected.append(gold_score_selected)
        
        # Calculate average gold score for selected outputs
        gold_scores_selected_tensor = torch.tensor(gold_scores_selected, dtype=torch.float32)
        avg_gold_reward_selected = gold_scores_selected_tensor.mean().item()
        
        print(f"Results for n={n}:")
        print(f"  Average gold reward (selected): {avg_gold_reward_selected:.4f}")
        
        result = {
            "user": args.name,
            "n": n,
            "training_size": args.training_size,
            "lambda": 0.01,  # Using lambda=0.01 for reward model
            "avg_gold_selected": avg_gold_reward_selected
        }
        results.append(result)
        
        # Save intermediate results
        with open("../../results/drift_bon.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n✅ Results saved to ../../results/drift_bon.jsonl")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'n':<10} {'Avg Gold Selected':<20}")
    print("-" * 30)
    for r in results:
        print(f"{r['n']:<10} {r['avg_gold_selected']:<20.4f}")