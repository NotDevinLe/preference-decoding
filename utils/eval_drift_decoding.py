import torch
import numpy as np
import json
import sys
import os
import argparse

# Add LLaMA-Factory to path
sys.path.append("LLaMA-Factory/src")

from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, default="user1", help="User name for the reward model")
parser.add_argument("--gold_model_path", type=str, required=True, help="Path to gold reward model")
parser.add_argument("--results_path", type=str, default="../results/drift_decoding_results.jsonl", help="Path to save results")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load gold reward model

base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print(f"Loading gold reward model from: {args.gold_model_path}")
gold_model_args = ModelArguments(
    model_name_or_path=base_model_path,  # Base model for gold
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

def format_llama3_prompt(prompt: str, response: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
    )

def get_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits, _, values = model(**inputs)
        return values[:, -1]

sample_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

for sample_size in sample_sizes:
    with open(f"../results/drift_decoding_responses/{args.user}_sample{sample_size}.json", "r") as f:
        data = json.load(f)

    print(f"Total data points available: {len(data)}")

    # Evaluate entries with gold model
    results = []
    gold_scores = []
    
    for i, entry in enumerate(data):
        prompt = entry['prompt']
        response = entry['response']

        # Get gold model scores
        formatted_chosen = format_llama3_prompt(prompt, response)

        gold_score = get_score(gold_model, gold_tokenizer, formatted_chosen)[0].item()

        gold_scores.append(gold_score)

        results.append({
            'index': i,
            'prompt': prompt,
            'gold_score': gold_score,
        })


    # Calculate mean gold score
    avg_gold_score = np.mean(gold_scores)

    # Save results with only gold scores
    results_path = args.results_path
    with open(results_path, "a") as f:
        f.write(json.dumps({
            "user": args.user,
            "n": sample_size,
            "avg_gold_score": avg_gold_score
        }) + "\n")
    print(f"Results saved to {results_path}")
    print(f"Sample size {sample_size}: Avg gold score = {avg_gold_score:.4f}")