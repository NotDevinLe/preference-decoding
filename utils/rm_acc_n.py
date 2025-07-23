import torch
import numpy as np
import json
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="User name (e.g., user1)")
parser.add_argument("--n", type=int, required=True, help="Number of samples the reward model was trained on")
args = parser.parse_args()

# Add LLaMA-Factory to path
sys.path.append("LLaMA-Factory/src")

from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your saved reward model (LoRA adapter)
adapter_path = f"/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/utils/saves/{args.name}_samples/{args.name}_{args.n}/reward"  # LoRA adapter path (relative to utils directory)
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

# Load test data
test_data_path = f"../data/preference/{args.name}_test.json"
print(f"Loading test data from: {test_data_path}")

if not os.path.exists(test_data_path):
    print(f"❌ Test data not found: {test_data_path}")
    sys.exit(1)

with open(test_data_path, "r") as f:
    data = json.load(f)

print(f"Total test data points: {len(data)}")

# Evaluate entries
results = []
correct_count = 0
total_count = len(data)

print(f"\nEvaluating {total_count} test examples...")
print("-" * 80)

for i, entry in enumerate(data):
    prompt = entry['prompt']
    chosen = entry['chosen']
    rejected = entry['rejected']

    # Format for reward model
    formatted_chosen = format_llama3_prompt(prompt, chosen)
    formatted_rejected = format_llama3_prompt(prompt, rejected)

    # Get scores
    chosen_score = get_score(model, tokenizer, formatted_chosen)[0].item()
    rejected_score = get_score(model, tokenizer, formatted_rejected)[0].item()
    
    # Determine if correctly ranked
    correctly_ranked = chosen_score > rejected_score
    if correctly_ranked:
        correct_count += 1

    # Store result
    result = {
        'index': i,
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'chosen_score': chosen_score,
        'rejected_score': rejected_score,
        'score_diff': chosen_score - rejected_score,
        'correctly_ranked': correctly_ranked
    }
    results.append(result)

    # Print progress
    if (i + 1) % 10 == 0 or i == total_count - 1:
        current_acc = correct_count / (i + 1)
        print(f"Progress: {i+1}/{total_count} | Current accuracy: {current_acc:.3f}")

# Calculate final statistics
accuracy = correct_count / total_count
avg_score_diff = np.mean([r['score_diff'] for r in results])
avg_chosen_score = np.mean([r['chosen_score'] for r in results])
avg_rejected_score = np.mean([r['rejected_score'] for r in results])
std_score_diff = np.std([r['score_diff'] for r in results])

print(f"\n{'='*80}")
print(f"FINAL EVALUATION RESULTS")
print(f"{'='*80}")
print(f"Model: {adapter_path}")
print(f"Test data: {test_data_path}")
print(f"Total test examples: {total_count}")
print(f"Correctly ranked: {correct_count}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Average score difference (chosen - rejected): {avg_score_diff:.4f} ± {std_score_diff:.4f}")
print(f"Average chosen score: {avg_chosen_score:.4f}")
print(f"Average rejected score: {avg_rejected_score:.4f}")
print(f"{'='*80}")

with open(f"../results/reward_results.jsonl", "a") as f:
    f.write(json.dumps({
        "name": args.name,
        "n": args.n,
        "accuracy": accuracy,
    }) + "\n")