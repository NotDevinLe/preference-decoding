import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
sys.path.append("LLaMA-Factory/src")
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

def format_llama3_prompt(prompt: str, response: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
    )

def get_score(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits, _, values = model(**inputs)
        return values[:, -1]

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--training_size", type=int, required=True, 
                        help="Number of training samples used to train the reward model")
    parser.add_argument("--n_values", type=str, default="5,10,15,20,50,100,150,200",
                        help="Comma-separated list of n values for best-of-n sampling")
    parser.add_argument("--gold_cache", type=str, default="../../results/gold_scores_bon.jsonl")
    args = parser.parse_args()

    # Load bon outputs
    data_path = "../../data/bon.json"
    with open(data_path, "r") as f:
        bon_data = json.load(f)
    print(f"Loaded {len(bon_data)} prompts from {data_path}")
    print(f"Each prompt has {len(bon_data[0]['outputs'])} outputs")

    # Load gold reward cache
    gold_cache = {}
    with open(args.gold_cache, "r") as f:
        for line in f:
            entry = json.loads(line)
            gold_cache[entry["prompt"]] = entry
    print(f"Loaded gold reward cache for {len(gold_cache)} prompts from {args.gold_cache}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load user reward model
    rm_path = f"../saves/normal/{args.name}_1b/toy_reward_{args.training_size}"
    base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading user reward model from: {rm_path}")
    model_args = ModelArguments(
        model_name_or_path=base_model_path,
        adapter_name_or_path=rm_path,
        trust_remote_code=True,
        use_fast_tokenizer=True,
    )
    finetuning_args = FinetuningArguments(stage="rm")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    user_model = load_model(
        tokenizer=tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=False,
        add_valuehead=True
    )
    user_model.to(device)
    user_model.eval()

    # Parse n values for best-of-n sampling
    n_values = [int(x) for x in args.n_values.split(",")]
    print(f"\n=== EVALUATING BEST-OF-N FOR N VALUES: {n_values} ===")

    results = []

    for n in n_values:
        print(f"\n--- Evaluating with n={n} outputs ---")
        
        # For each prompt, select the output with the highest user reward
        gold_scores_selected = []
        
        for item in bon_data:
            prompt = item["prompt"]
            # Only use first n outputs for best-of-n
            outputs = item["outputs"][:n]
            
            # Score outputs with user reward model
            user_scores = []
            for output in outputs:
                formatted = format_llama3_prompt(prompt, output)
                score = get_score(user_model, tokenizer, formatted, device)
                user_scores.append(score[0].item() if score.dim() > 0 else score.item())
            
            # Select best according to user model
            idx = int(np.argmax(user_scores))
            
            # Get gold score for selected output
            gold_entry = gold_cache[prompt]
            gold_scores_subset = gold_entry["output_scores"][:n]
            gold_score_selected = gold_scores_subset[idx]
            gold_scores_selected.append(gold_score_selected)
        
        avg_gold_reward_selected = float(np.mean(gold_scores_selected))
        print(f"Results for n={n}:")
        print(f"  Average gold reward (selected): {avg_gold_reward_selected:.4f}")
        
        result = {
            "user": args.name,
            "n": n,
            "training_size": args.training_size,
            "avg_gold_selected": avg_gold_reward_selected
        }
        results.append(result)
        
        # Save intermediate results
        with open("../../results/rm_bon_1b.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n✅ Results saved to ../../results/rm_bon_1b.jsonl")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'n':<10} {'Avg Gold Selected':<20}")
    print("-" * 30)
    for r in results:
        print(f"{r['n']:<10} {r['avg_gold_selected']:<20.4f}")