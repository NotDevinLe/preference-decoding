import pickle
import argparse
import numpy as np
import torch
import json
from drift import get_training_matrix
import vllm
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, persona_prompts, user1_reg_prompts, user2_reg_prompts, user4_reg_prompts, base_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='User name (e.g., user1)')
parser.add_argument('--samples', type=int, default=200, help='Maximum number of samples to use')
parser.add_argument('--save_path', type=str, default="../results/user_p.jsonl", help='Path to save results')
args = parser.parse_args()

# Load user data from JSON format
data_path = f"../data/preference/{args.name}_train.json"
print(f"Loading data from: {data_path}")

with open(data_path, "r") as f:
    preference_data = json.load(f)

print(f"Loaded {len(preference_data)} preference pairs")

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

data = []
for j in range(args.samples):
    question = preference_data[j]['prompt']
    yw = preference_data[j]['chosen']  # winning/chosen response
    yl = preference_data[j]['rejected']  # losing/rejected response
    data.append((question, yw, yl))

print(f"Converted {len(data)} samples to drift format")

d = get_training_matrix(data, model, tokenizer, base_prompt, attribute_prompts, device)

sample_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

for sample_size in sample_sizes:
    curr = d[:sample_size].cpu().numpy()
    curr = np.mean(curr, axis=0)
    if np.linalg.norm(curr, ord=1) > 1:
        curr = curr * (1 / np.linalg.norm(curr, ord=1))
    
    adding = {'user': args.name, 'sample_size': sample_size, 'p': curr.tolist()}
    with open("../results/user_p.jsonl", "a") as f:
        f.write(json.dumps(adding) + "\n")
