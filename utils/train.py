import pickle
import argparse
import numpy as np
import torch
import json
from drift import get_training_matrix
import vllm
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, persona_prompts, user1_reg_prompts, user2_reg_prompts, user4_reg_prompts, base_prompt
import cvxpy as cp

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='User name (e.g., user1)')
parser.add_argument('--save_path', type=str, default="../results/user_p.jsonl", help='Path to save results')
args = parser.parse_args()

# Load user data from JSON format
data_path = f"../data/preference/{args.name}_train.json"
print(f"Loading data from: {data_path}")

with open(data_path, "r") as f:
    preference_data = json.load(f)

print(f"Loaded {len(preference_data)} preference pairs from {data_path}")

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=8192)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

data = []
for j in range(len(preference_data)):
    question = preference_data[j]['prompt']
    yw = preference_data[j]['chosen']  # winning/chosen response
    yl = preference_data[j]['rejected']  # losing/rejected response
    data.append((question, yw, yl))

print(f"Converted {len(data)} samples to drift format")

lambdas = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]

d = get_training_matrix(data[:200], model, tokenizer, base_prompt, attribute_prompts, device)

for lambda_ in lambdas:
    sample_sizes = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

    for sample_size in sample_sizes:
        curr = d[:sample_size]
        # Take mean over samples and convert to numpy
        d_mean = torch.mean(curr, dim=0).cpu().numpy()
        
        p_var = cp.Variable(len(d_mean))  # Use number of attributes, not samples
        # Remove redundant constraint since you're using L1 penalty
        
        linear_term = d_mean @ -p_var  # Use numpy array
        l1_penalty = lambda_ * cp.norm1(p_var)
        l2_penalty = 0.01 * cp.sum_squares(p_var)  # Use smaller L2 penalty
        objective = cp.Minimize(linear_term + l1_penalty + l2_penalty)
        problem = cp.Problem(objective)  # No constraints
        problem.solve()

        if p_var.value is None:
            print("Optimization failed, falling back to simple normalization")
            # Use the current sample mean, not full d
            current_norm = torch.norm(torch.mean(curr, dim=0), p=1)
            if current_norm > 1:
                p = torch.mean(curr, dim=0) * (1 / current_norm)
            else:
                p = torch.mean(curr, dim=0)
            p = p.cpu().numpy()
        else:
            p = p_var.value  # Use the optimization result

        adding = {'user': args.name, 'sample_size': sample_size, 'p': p.tolist(), 'lambda': lambda_}
        with open(args.save_path, "a") as f:
            f.write(json.dumps(adding) + "\n")