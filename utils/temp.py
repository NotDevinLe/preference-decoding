import numpy as np
import random
import itertools
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
from drift import get_training_matrix, l1_solve
import torch
import json
import matplotlib.pyplot as plt
from attribute_prompts import base_prompt

selected_prompts = [
    "You are an AI assistant that speaks like a pirate.",
    "You are an AI assistant that speaks like a cowboy.",
    "You are an AI assistant that speaks in internet slang.",
    "You are an AI assistant that speaks like a robot.",
    "You are an AI assistant that speaks like a university professor."
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=16384
)

data_path = '../data/toy/attribute/user1_train.json'

with open(data_path, 'r') as f:
    data = json.load(f)

print(f"Data loaded from {data_path}")

data = data[:200]

d = get_training_matrix(data, llm, tokenizer, base_prompt, selected_prompts, device)

d_mean = torch.mean(d, dim=0).cpu().numpy()

lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 100, 1000]
# lambdas = np.linspace(1, 10, 20)

p_list = []
found = False
for l1_lambda in lambdas:
    p = l1_solve(d_mean, l1_lambda)
    if sum(p <= 1e-5) == len(selected_prompts) - 2 and not found:
        found = True
        print(p / np.linalg.norm(p, ord=1))
    p_list.append(p)

p_array = np.array(p_list)  # Shape: (num_lambdas, num_attributes)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each attribute's weight as a function of lambda
for i in range(len(selected_prompts)):
    plt.plot(lambdas, p_array[:, i], 'o-', linewidth=2, markersize=6, 
             label=f'Attr {i}: {selected_prompts[i][:20]}...')

plt.xlabel('L1 Lambda', fontsize=12)
plt.ylabel('P Weight', fontsize=12)
plt.title('Attribute Weights vs L1 Regularization Strength\n(L2 Lambda = 1)', fontsize=14)
plt.xscale('log')  # Log scale for lambda since values span orders of magnitude
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

# Add horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.savefig('p_weights_vs_lambda.png', dpi=300, bbox_inches='tight')
plt.show()

# # Print the actual values for inspection
# print("Lambda\t", end="")
# for i, prompt in enumerate(selected_prompts):
#     print(f"Attr{i}\t\t", end="")
# print()

# for i, lam in enumerate(lambdas):
#     print(f"{lam}\t", end="")
#     for j in range(len(selected_prompts)):
#         print(f"{p_array[i, j]:.4f}\t\t", end="")
#     print()

# Also create a separate plot showing the L1 norm progression
plt.figure(figsize=(10, 6))
l1_norms = [np.linalg.norm(p, ord=1) for p in p_list]
plt.plot(lambdas, l1_norms, 'ro-', linewidth=2, markersize=8)
plt.xlabel('L1 Lambda', fontsize=12)
plt.ylabel('L1 Norm of P Vector', fontsize=12)
plt.title('L1 Norm vs L1 Regularization Strength', fontsize=14)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l1_norm_vs_lambda.png', dpi=300, bbox_inches='tight')
plt.show()

# print(f"\nL1 Norms:")
# for i, (lam, norm) in enumerate(zip(lambdas, l1_norms)):
#     print(f"Lambda {lam}: L1 norm = {norm:.4f}")