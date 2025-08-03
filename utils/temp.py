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
from attribute_prompts import base_prompt, attribute_prompts

with open("../data/preference/user1_train.json", "r") as f:
    data = json.load(f)

data = data[:200]
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

# Convert from dict format to tuple format
data = [
    (item["prompt"], item["chosen"], item["rejected"]) 
    for item in data
]


d = get_training_matrix(data, llm, tokenizer, base_prompt, attribute_prompts, device)

d_mean = torch.mean(d, dim=0).cpu().numpy()

lambdas = np.linspace(0,0.1,100)

p_list = []
found = False
for l1_lambda in lambdas:
    p = l1_solve(d_mean, l1_lambda)
    if sum(p <= 1e-5) == len(attribute_prompts) - 2 and not found:
        found = True
        print(p / np.linalg.norm(p, ord=1))
    p_list.append(p)

p_array = np.array(p_list)  # Shape: (num_lambdas, num_attributes)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each attribute's weight as a function of lambda
for i in range(len(attribute_prompts)):
    plt.plot(lambdas, p_array[:, i], 'o-', linewidth=2, markersize=6, 
             label=f'Attr {i}: {attribute_prompts[i][:20]}...')

plt.xlabel('L1 Lambda', fontsize=12)
plt.ylabel('P Weight', fontsize=12)
plt.title('Attribute Weights vs L1 Regularization Strength\n(Preference)', fontsize=14)
plt.xscale('log')  # Log scale for lambda since values span orders of magnitude
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()

# Add horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.savefig('p_weights_vs_lambda.png', dpi=300, bbox_inches='tight')
plt.show()