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
parser.add_argument("--name", type=str, default="user11", required=True)
parser.add_argument("--n", type=int, default=150)
args = parser.parse_args()

# Path to your saved reward model (LoRA adapter)
adapter_path = f"/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/utils/saves/normal/{args.name}_1b/rm"  # LoRA adapter path (relative to utils directory)
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Base model from adapter config

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
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False)

def get_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits, _, values = model(**inputs)
        return values[:, -1]

# Load the preference data
print("Loading BON data...")
with open(f"../data/bon_attributes.json", "r") as f:
    data = json.load(f)

print(f"Total data points available: {len(data)}")

# Evaluate entries
selected_indices = []
n_candidates = None
for i, entry in enumerate(data):
    prompt = entry['prompt']
    outputs = entry['outputs']

    if n_candidates is None:
        n_candidates = len(outputs)

    scores = np.array([])

    for output in outputs:
        formatted = format_llama3_prompt(prompt, output)
        score = get_score(model, tokenizer, formatted)
        scores = np.append(scores, score[0].item() if score.dim() > 0 else score.item())

    chosen_idx = int(np.argmax(scores))
    selected_indices.append(chosen_idx)

# Save results in the requested JSONL format
results_dir = "../results/rm_bon_responses"
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"{args.name}.jsonl")

with open(results_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": n_candidates,
        "training_size": args.n,
        "selected_indices": selected_indices,
    }) + "\n")

print(f"Results saved to {results_path}")