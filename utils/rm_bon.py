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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
args = parser.parse_args()

# Load bon outputs
data_path = "../data/bon.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)
bon_data = bon_data[:500]
print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Load gold reward cache
gold_cache = {}
with open(args.gold_cache, "r") as f:
    for line in f:
        entry = json.loads(line)
        gold_cache[entry["prompt"]] = entry
print(f"Loaded gold reward cache for {len(gold_cache)} prompts from {args.gold_cache}")

# Load user reward model
rm_path = f"saves/normal/{args.name}/toy_reward_{args.sample_size}"
base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print(f"Loading user reward model from: {rm_path}")
model_args = ModelArguments(
    model_name_or_path=base_model_path,
    adapter_name_or_path=rm_path,
    trust_remote_code=True,
    use_fast_tokenizer=True,
)
finetuning_args = FinetuningArguments(stage="rm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# For each prompt, select the output with the highest user reward
selected_outputs = []
gold_scores_selected = []
gold_scores_all = []
selected_minus_max = []
print("Scoring outputs with user reward model and using gold cache...")
for item in bon_data:
    prompt = item["prompt"]
    outputs = item["outputs"]
    user_scores = []
    for output in outputs:
        formatted = format_llama3_prompt(prompt, output)
        score = get_score(user_model, tokenizer, formatted)[0].item()
        user_scores.append(score)
    idx = int(np.argmax(user_scores))
    selected_outputs.append({
        "prompt": prompt,
        "output": outputs[idx],
        "score": user_scores[idx]
    })
    # Use gold cache for this prompt
    gold_entry = gold_cache[prompt]
    gold_score_selected = gold_entry["output_scores"][idx]
    gold_scores_selected.append(gold_score_selected)
    gold_scores_all.append(np.mean(gold_entry["output_scores"]))
    max_gold = max(gold_entry["output_scores"])
    selected_minus_max.append(max_gold - gold_score_selected)
print(f"Selected best outputs for {len(selected_outputs)} prompts.")

avg_gold_reward_selected = float(np.mean(gold_scores_selected))
avg_gold_reward_all = float(np.mean(gold_scores_all))
uplift = avg_gold_reward_selected - avg_gold_reward_all
avg_selected_minus_max = float(np.mean(selected_minus_max))
print(f"Average gold reward (selected): {avg_gold_reward_selected:.4f}")
print(f"Average gold reward (all): {avg_gold_reward_all:.4f}")
print(f"Uplift (selected - all): {uplift:.4f}")
print(f"max gold RM - Average (selected): {avg_selected_minus_max:.4f}")

# Save results in the required format
with open("../results/rm_bon.jsonl", "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": args.sample_size,
        "uplift": uplift,
        "selected_minus_max": avg_selected_minus_max
    }) + "\n")
print(f"✅ Results saved to ../results/rm_bon.jsonl")
