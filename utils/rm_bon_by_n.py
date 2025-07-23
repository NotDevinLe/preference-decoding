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
parser.add_argument("--gold_cache", type=str, default="../results/gold_scores.jsonl")
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

# Load user reward model
rm_path = f"saves/normal/{args.name}/toy_reward_200"
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

max_k = 20
results_by_k = []
for k in range(2, max_k + 1, 2):
    selected_gold_scores = []
    all_gold_scores = []
    selected_minus_max = []
    for item in bon_data:
        prompt = item["prompt"]
        outputs = item["outputs"][:k]
        user_scores = []
        for output in outputs:
            formatted = format_llama3_prompt(prompt, output)
            score = get_score(user_model, tokenizer, formatted)[0].item()
            user_scores.append(score)
        idx = int(np.argmax(user_scores))
        gold_entry = gold_cache[prompt]
        gold_scores_k = gold_entry["output_scores"][:k]
        gold_score_selected = gold_scores_k[idx]
        max_gold_at_k = max(gold_scores_k)
        selected_gold_scores.append(gold_score_selected)
        all_gold_scores.append(np.mean(gold_scores_k))
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
with open("../results/rm_bon_by_n.jsonl", "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print(f"✅ Results saved to ../results/rm_bon_by_n.jsonl")
