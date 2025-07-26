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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1", required=True)
parser.add_argument("--n", type=int, default=10, required=True)
args = parser.parse_args()

# Path to your saved reward model (LoRA adapter)
adapter_path = f"/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/utils/saves/normal/{args.name}/toy_reward"  # LoRA adapter path (relative to utils directory)
base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Base model from adapter config

print("Loading model...")
# Setup arguments for LLaMA-Factory
model_args = ModelArguments(
    model_name_or_path=base_model_path,  # Base model path
    adapter_name_or_path=adapter_path,   # LoRA adapter path
    trust_remote_code=True,
    use_fast_tokenizer=True,
)

finetuning_args = FinetuningArguments(
    stage="rm"  # this will activate value head logic
)

# Load tokenizer and model properly using LLaMA-Factory
print("Loading tokenizer...")
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]

# Set padding token - try different approaches
if tokenizer.pad_token is None:
    # Method 1: Use EOS token as padding token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Method 2: If you want to use a specific token, you can also do:
    # tokenizer.pad_token = "<pad>"
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    # Method 3: For some models, you might need to resize embeddings
    # if hasattr(model, 'resize_token_embeddings'):
    #     model.resize_token_embeddings(len(tokenizer))

print("Loading reward model...")
model = load_model(
    tokenizer=tokenizer,
    model_args=model_args,
    finetuning_args=finetuning_args,
    is_trainable=False,
    add_valuehead=True
)

model.to(device)
model.eval()

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

# Load the preference data
print("Loading preference data...")
with open("../data/preference/user1_test.json", "r") as f:
    data = json.load(f)

print(f"Total data points available: {len(data)}")

# Evaluate entries
results = []
for i, entry in enumerate(data):
    prompt = entry['prompt']
    chosen = entry['chosen']
    rejected = entry['rejected']

    formatted_chosen = format_llama3_prompt(prompt, chosen)
    formatted_rejected = format_llama3_prompt(prompt, rejected)

    chosen_score = get_score(model, tokenizer, formatted_chosen)[0].item()
    rejected_score = get_score(model, tokenizer, formatted_rejected)[0].item()

    results.append({
        'index': i,
        'prompt': prompt,
        'chosen_score': chosen_score,
        'rejected_score': rejected_score,
        'score_diff': chosen_score - rejected_score,
        'correctly_ranked': chosen_score > rejected_score
    })

    print(f"Entry {i+1}: Chosen={chosen_score:.3f}, Rejected={rejected_score:.3f}, Diff={chosen_score-rejected_score:.3f}, Correct={chosen_score > rejected_score}")

# Summary stats
correct_rankings = sum(1 for r in results if r['correctly_ranked'])
accuracy = correct_rankings / len(results)
avg_score_diff = np.mean([r['score_diff'] for r in results])
avg_chosen_score = np.mean([r['chosen_score'] for r in results])
avg_rejected_score = np.mean([r['rejected_score'] for r in results])

# Save results in the same format as approximation_results.jsonl
results_path = "../results/reward_results.jsonl"
with open(results_path, "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": args.n,
        "acc": accuracy,
    }) + "\n")
print(f"Results saved to {results_path}")