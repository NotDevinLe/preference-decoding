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
    from utils.attribute_prompts import attribute_prompts, base_prompt
except ImportError:
    sys.path.append('..')
    from drift import get_scores
    from attribute_prompts import attribute_prompts, base_prompt

# Import LLaMA-Factory modules for loading trained models
from llamafactory.hparams import ModelArguments, FinetuningArguments
from llamafactory.model import load_tokenizer, load_model

def format_llama3_prompt(prompt: str, response: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
    )

def get_gold_score(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        _, _, values = model(**inputs)
        return values[:, -1]

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="user1")
    parser.add_argument("--gold_model_path", type=str, required=True, help="Path to gold reward model")
    parser.add_argument("--gold_cache", type=str, default="../../results/gold_scores_bon.jsonl")
    parser.add_argument("--generate_gold", action="store_true", 
                        help="Generate gold scores for first 20 outputs")
    parser.add_argument("--gold_only", action="store_true",
                        help="Only generate gold scores, skip reward model evaluation")
    parser.add_argument("--k_values", type=str, default="2,4,6,8,10,12,14,16,18,20",
                        help="Comma-separated list of k values to evaluate")
    parser.add_argument("--p_path", type=str, default="../../results/preference/user1_p.json",
                        help="Path to JSONL file containing p vectors")
    parser.add_argument("--training_size", type=int, default=200,
                        help="Training size to select p vector for")
    args = parser.parse_args()

    # Load bon outputs
    data_path = "../../data/bon.json"
    with open(data_path, "r") as f:
        bon_data = json.load(f)

    print(f"Loaded {len(bon_data)} prompts from {data_path}")
    print(f"Each prompt has {len(bon_data[0]['outputs'])} outputs")

    # Setup device and model
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
        
        # Get max k value to determine how many outputs to score
        k_values = [int(x) for x in args.k_values.split(",")]
        max_k = max(k_values)
        
        # Generate gold scores for first max_k outputs of each prompt
        print(f"Generating gold scores for first {max_k} outputs per prompt using trained gold model...")
        gold_cache = {}
        
        for i, item in enumerate(bon_data):
            prompt = item["prompt"]
            # Only score first max_k outputs to save time
            outputs_to_score = item["outputs"][:max_k]
            
            # Get gold scores using the trained reward model directly
            scores = []
            for output in outputs_to_score:
                formatted_text = format_llama3_prompt(prompt, output)
                score = get_gold_score(gold_model, gold_tokenizer, formatted_text, device)
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

    # Setup reward model
    print("\n=== SETTING UP REWARD MODEL ===")
    base_model_path = "meta-llama/Llama-3.2-1B-Instruct"

    # Initialize tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = LLM(
        model=base_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # Load p vector from JSONL file
    print(f"Loading p vector from: {args.p_path}")
    p = None
    with open(args.p_path, "r") as f:
        p_list = json.load(f)
        for entry in p_list:
            if entry["sample_size"] == args.training_size:
                p = entry["p"]
                break
        print(f"Found p vector for training_size={args.training_size}")
    
    if p is None:
        raise ValueError(f"No p vector found for training_size={args.training_size} in {args.p_path}")
    
    p = np.array(p)
    p = p / np.linalg.norm(p)
    p = p.tolist()

    # Sparsify p using torch operations for consistency
    p_tensor = torch.tensor(p, device=device)
    abs_p = torch.abs(p_tensor)
    topk_values, topk_idx = torch.topk(abs_p, k=min(7, len(p)))  # Handle case where p has < 7 elements

    p_sparse = torch.zeros_like(p_tensor)
    p_sparse[topk_idx] = p_tensor[topk_idx]
    p_sparse_np = p_sparse.cpu().numpy()

    print(f"Sparsified p: kept top {len(topk_idx)} elements")
    print(f"Non-zero elements: {torch.count_nonzero(p_sparse).item()}")

    # Choose the right attribute prompts
    attribute_list = attribute_prompts

    print(f"Using {len(attribute_list)} attribute prompts")

    # Parse k values and get max k for precomputation
    k_values = [int(x) for x in args.k_values.split(",")]
    max_k = max(k_values)
    print(f"Will evaluate k values: {k_values}")

    # Precompute all scores for all outputs (up to max_k)
    print(f"Computing drift scores for all outputs up to {max_k}...")
    all_prompt_outputs = [(item["prompt"], item["outputs"][:max_k]) for item in bon_data]
    all_scores_full = get_scores(
        all_prompt_outputs,
        model, p_sparse_np, base_prompt, attribute_list, device, tokenizer
    )

    print(f"Computed scores for {len(all_scores_full)} prompts")

    # Evaluate for different values of k
    results_by_k = []
    for k in k_values:
        print(f"Evaluating for k={k}...")
        selected_gold_scores = []
        
        for item, scores in zip(bon_data, all_scores_full):
            # Use only the first k outputs and their scores
            scores_k = scores[:k]
            
            # Find best output according to our model
            if isinstance(scores_k, torch.Tensor):
                idx = int(torch.argmax(scores_k).cpu())  # Use torch.argmax and move to CPU
            else:
                idx = int(np.argmax(scores_k))  # Use numpy if it's already a numpy array
            
            # Get gold score for selected output from cache
            gold_entry = gold_cache[item["prompt"]]
            gold_scores_subset = gold_entry["output_scores"][:k]
            gold_score_selected = gold_scores_subset[idx]
            
            selected_gold_scores.append(gold_score_selected)
        
        # Calculate average selected gold score
        avg_selected = float(np.mean(selected_gold_scores))
        
        results_by_k.append({
            "user": args.name,
            "k": k,
            "training_size": args.training_size,
            "avg_selected_gold": avg_selected,
        })
        
        print(f"k={k}: avg_selected_gold={avg_selected:.4f}")

    # Save results
    output_file = f"../../results/bon_by_n/drift_bon_by_n.jsonl"
    with open(output_file, "a") as f:
        for res in results_by_k:
            f.write(json.dumps(res) + "\n")
    print(f"✅ Results saved to {output_file}")

    # Print summary
    best_k = max(results_by_k, key=lambda x: x['avg_selected_gold'])
    print(f"\nBest performance at k={best_k['k']} with avg_selected_gold={best_k['avg_selected_gold']:.4f}")