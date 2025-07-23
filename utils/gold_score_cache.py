import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM
import numpy as np
sys.path.append("LLaMA-Factory/src")
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--output_path", type=str, default="../results/gold_scores.jsonl")
parser.add_argument("--bon_path", type=str, default="../data/bon_200.json")
parser.add_argument("--max_prompts", type=int, default=None, help="Limit number of prompts (for debugging)")
args = parser.parse_args()

# Load bon outputs
data_path = args.bon_path
with open(data_path, "r") as f:
    bon_data = json.load(f)
if args.max_prompts is not None:
    bon_data = bon_data[:args.max_prompts]
print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Load gold reward model
base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
gold_path = f"saves/golden/{args.name}/toy_reward"
print(f"Loading gold reward model from: {gold_path}")
model_args = ModelArguments(
    model_name_or_path=base_model_path,
    adapter_name_or_path=gold_path,
    trust_remote_code=True,
    use_fast_tokenizer=True,
)
finetuning_args = FinetuningArguments(stage="rm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gold_model = load_model(
    tokenizer=tokenizer,
    model_args=model_args,
    finetuning_args=finetuning_args,
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

# Score all outputs for each prompt and save
with open(args.output_path, "w") as fout:
    for idx, item in enumerate(bon_data):
        prompt = item["prompt"]
        outputs = item["outputs"]
        scores = []
        for output in outputs:
            formatted = format_llama3_prompt(prompt, output)
            score = get_score(gold_model, tokenizer, formatted)[0].item()
            scores.append(score)
        max_reward = float(np.max(scores))
        fout.write(json.dumps({
            "prompt": prompt,
            "output_scores": scores,
            "max_reward": max_reward
        }) + "\n")
        print(f"Processed prompt {idx+1}/{len(bon_data)} (max_reward={max_reward:.4f})")
print(f"✅ Gold reward scores saved to {args.output_path}") 