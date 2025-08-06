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
parser.add_argument("--gold_model_path", type=str, required=True, help="Path to gold reward model")
args = parser.parse_args()

# Load bon outputs
data_path = "../data/bon.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

# Load gold reward model
print(f"Loading gold reward model from: {args.gold_model_path}")
gold_model_args = ModelArguments(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    adapter_name_or_path=args.gold_model_path,
    trust_remote_code=True,
    use_fast_tokenizer=True,
)
gold_finetuning_args = FinetuningArguments(stage="rm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load user reward model
rm_path = f"saves/normal/{args.name}_1b/toy_reward_200"
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

k_list = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
results_by_k = []
for k in k_list:
    selected_gold_scores = []
    for item in bon_data:
        prompt = item["prompt"]
        outputs = item["outputs"][:k]
        user_scores = []
        for output in outputs:
            formatted = format_llama3_prompt(prompt, output)
            score = get_score(user_model, tokenizer, formatted)[0].item()
            user_scores.append(score)
        idx = int(np.argmax(user_scores))
        selected_output = outputs[idx]
        
        # Score the selected output with gold model
        formatted_selected = format_llama3_prompt(prompt, selected_output)
        gold_score_selected = get_score(gold_model, gold_tokenizer, formatted_selected)[0].item()
        selected_gold_scores.append(gold_score_selected)
    
    avg_selected_gold = float(np.mean(selected_gold_scores))
    results_by_k.append({
        "user": args.name,
        "k": k,
        "avg_selected_gold": avg_selected_gold
    })
    print(f"k={k}: avg_selected_gold={avg_selected_gold:.4f}")

# Save results
with open("../results/rm_bon_by_n.jsonl", "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print(f"✅ Results saved to ../results/rm_bon_by_n.jsonl")
