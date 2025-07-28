import numpy as np
import random
import itertools
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
import cvxpy as cp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.drift import get_training_matrix
    from utils.attribute_prompts import attribute_prompts, base_prompt
except ImportError:
    # If running from evals folder directly
    sys.path.append('..')
    from drift import get_training_matrix
    from attribute_prompts import attribute_prompts, base_prompt
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid search parameters
l1_lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
l2_lambdas = [0.001, 0.01, 0.1, 0.5, 1.0]

def elastic_net_solve(d_mean, l1_lambda, l2_lambda):
    """Solve elastic net optimization problem"""
    p_var = cp.Variable(len(d_mean))
    
    linear_term = d_mean @ -p_var
    l1_penalty = l1_lambda * cp.norm1(p_var)
    l2_penalty = l2_lambda * cp.sum_squares(p_var)
    objective = cp.Minimize(linear_term + l1_penalty + l2_penalty)
    
    problem = cp.Problem(objective)
    problem.solve()
    
    if p_var.value is None:
        print(f"Optimization failed for L1={l1_lambda}, L2={l2_lambda}, using normalization fallback")
        # Use simple normalization as fallback
        current_norm = np.linalg.norm(d_mean, ord=1)
        if current_norm > 1:
            p = d_mean * (1 / current_norm)
        else:
            p = d_mean.copy()
        return p, "fallback", None
    else:
        return p_var.value, problem.status, problem.value

def generate_data(system_prompt1, system_prompt2, base_prompt, prob1, prob2, size, dolly_ds):
    base_prompt_inputs = []
    base_prompt_outputs = []

    instructions = [build_prompt(row["instruction"], row["context"]) for row in dolly_ds.shuffle().select(range(size))]

    for instruction in instructions:
        base_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        base_prompt_inputs.append(base_prompt_input)

    base_prompt_outputs = llm.generate(base_prompt_inputs, sampling_params)
    base_prompt_outputs = [output.outputs[0].text.strip() for output in base_prompt_outputs]

    attr1_prompt_inputs = []
    attr1_prompt_outputs = []

    for instruction in instructions[:int(len(instructions) * prob1)]:
        attr1_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt1},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        attr1_prompt_inputs.append(attr1_prompt_input)

    attr1_prompt_outputs = llm.generate(attr1_prompt_inputs, sampling_params)
    attr1_prompt_outputs = [output.outputs[0].text.strip() for output in attr1_prompt_outputs]

    attr2_prompt_inputs = []
    attr2_prompt_outputs = []

    for instruction in instructions[int(len(instructions) * prob1):]:
        attr2_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        attr2_prompt_inputs.append(attr2_prompt_input)

    attr2_prompt_outputs = llm.generate(attr2_prompt_inputs, sampling_params)
    attr2_prompt_outputs = [output.outputs[0].text.strip() for output in attr2_prompt_outputs]

    attribute_prompts_outputs = attr1_prompt_outputs + attr2_prompt_outputs

    all_data = []
    for i in range(len(instructions)):
        all_data.append({
            "prompt": instructions[i],
            "chosen": attribute_prompts_outputs[i],
            "rejected": base_prompt_outputs[i]
        })

    return all_data

selected_prompts = [
    "You are an AI assistant that speaks in Japanese.",
    "You are an AI assistant that speaks in French.",
    "You are an AI assistant that speaks in Spanish.",
    "You are an AI assistant that speaks in German.",
    "You are an AI assistant that speaks in Italian.",
]

# Model setup
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=16384
)

# Sampling configuration
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=4096,
    stop=[]
)

# Load Dolly dataset
dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

def build_prompt(instruction, context):
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction

# Run experiment on all pairs
results = []
train_size = 200

pairs = [(0,1)]

for experiment_idx, pair in enumerate(pairs):
    print(f"\n=== EXPERIMENT {experiment_idx + 1}: PAIR {pair} ===")
    
    # Randomly sample 2 attributes from the 5
    attr1, attr2 = selected_prompts[pair[0]], selected_prompts[pair[1]]
    
    # Assign random probabilities
    prob1 = random.random()
    prob2 = 1 - prob1
    
    print(f"Selected attributes:")
    print(f"  Attribute A: {attr1[:50]}... (prob: {prob1:.3f})")
    print(f"  Attribute B: {attr2[:50]}... (prob: {prob2:.3f})")
    
    # Generate training data
    train_data = generate_data(attr1, attr2, base_prompt, prob1, prob2, train_size, dolly_ds)
    print(f"Generated {len(train_data)} training samples")
    
    # Get training matrix
    training_matrix = get_training_matrix(
        [(item['prompt'], item['chosen'], item['rejected']) for item in train_data], 
        llm, tokenizer, base_prompt, selected_prompts, device
    )
    
    # Compute empirical mean (true distribution from data)
    d_mean_empirical = torch.mean(training_matrix, dim=0).cpu().numpy()
    
    # True theoretical distribution
    true_p = np.zeros(len(selected_prompts))
    attr1_idx = selected_prompts.index(attr1)
    attr2_idx = selected_prompts.index(attr2)
    true_p[attr1_idx] = prob1
    true_p[attr2_idx] = prob2
    
    print(f"True theoretical distribution: {true_p}")
    print(f"Empirical distribution: {d_mean_empirical}")
    
    # Grid search over lambda values
    total_lambda_combinations = len(l1_lambdas) * len(l2_lambdas)
    lambda_combination = 0
    
    for l1_lambda in l1_lambdas:
        for l2_lambda in l2_lambdas:
            lambda_combination += 1
            print(f"  Lambda combination {lambda_combination}/{total_lambda_combinations}: L1={l1_lambda}, L2={l2_lambda}")
            
            # Solve elastic net
            p_recovered, solver_status, objective_value = elastic_net_solve(d_mean_empirical, l1_lambda, l2_lambda)
            
            # Calculate recovery metrics (against true theoretical distribution)
            mse_true = np.mean((p_recovered - true_p) ** 2)
            mae_true = np.mean(np.abs(p_recovered - true_p))
            cosine_sim_true = np.dot(p_recovered, true_p) / (np.linalg.norm(p_recovered) * np.linalg.norm(true_p) + 1e-8)
            
            # Calculate recovery metrics (against empirical distribution)
            mse_emp = np.mean((p_recovered - d_mean_empirical) ** 2)
            mae_emp = np.mean(np.abs(p_recovered - d_mean_empirical))
            cosine_sim_emp = np.dot(p_recovered, d_mean_empirical) / (np.linalg.norm(p_recovered) * np.linalg.norm(d_mean_empirical) + 1e-8)
            
            # Check if top attributes are correctly identified
            top_2_recovered = np.argsort(np.abs(p_recovered))[-2:]
            top_2_true = np.argsort(np.abs(true_p))[-2:]
            correct_top_2 = len(set(top_2_recovered) & set(top_2_true)) / 2
            
            # Sparsity and regularization metrics
            sparsity = np.sum(np.abs(p_recovered) < 1e-6) / len(p_recovered)
            l1_norm = np.linalg.norm(p_recovered, ord=1)
            l2_norm = np.linalg.norm(p_recovered, ord=2)
            
            result = {
                'experiment': experiment_idx + 1,
                'pair': pair,
                'attr1': attr1,
                'attr2': attr2,
                'true_prob1': prob1,
                'true_prob2': prob2,
                'l1_lambda': l1_lambda,
                'l2_lambda': l2_lambda,
                'solver_status': str(solver_status),
                'objective_value': float(objective_value) if objective_value is not None else None,
                
                # Distributions
                'true_distribution': true_p.tolist(),
                'empirical_distribution': d_mean_empirical.tolist(),
                'recovered_distribution': p_recovered.tolist(),
                
                # Metrics vs true distribution
                'mse_true': mse_true,
                'mae_true': mae_true,
                'cosine_similarity_true': cosine_sim_true,
                
                # Metrics vs empirical distribution
                'mse_empirical': mse_emp,
                'mae_empirical': mae_emp,
                'cosine_similarity_empirical': cosine_sim_emp,
                
                # Other metrics
                'top_2_accuracy': correct_top_2,
                'sparsity': sparsity,
                'l1_norm': l1_norm,
                'l2_norm': l2_norm
            }
            
            results.append(result)

print(f"\n{'='*80}")
print("GRID SEARCH COMPLETED")
print(f"{'='*80}")
print(f"Total experiments: {len(pairs)}")
print(f"Lambda combinations per experiment: {len(l1_lambdas)} × {len(l2_lambdas)} = {len(l1_lambdas) * len(l2_lambdas)}")
print(f"Total results: {len(results)}")

# Find best lambda combinations across all experiments
print(f"\nBest lambda combinations (averaged across all experiments):")

# Group by lambda combination
lambda_results = {}
for result in results:
    key = (result['l1_lambda'], result['l2_lambda'])
    if key not in lambda_results:
        lambda_results[key] = []
    lambda_results[key].append(result)

# Calculate averages for each lambda combination
lambda_summary = []
for (l1, l2), group in lambda_results.items():
    avg_cosine_true = np.mean([r['cosine_similarity_true'] for r in group])
    avg_mse_true = np.mean([r['mse_true'] for r in group])
    avg_top2_acc = np.mean([r['top_2_accuracy'] for r in group])
    avg_sparsity = np.mean([r['sparsity'] for r in group])
    
    lambda_summary.append({
        'l1_lambda': l1,
        'l2_lambda': l2,
        'avg_cosine_similarity_true': avg_cosine_true,
        'avg_mse_true': avg_mse_true,
        'avg_top_2_accuracy': avg_top2_acc,
        'avg_sparsity': avg_sparsity
    })

# Sort by different metrics
best_cosine = max(lambda_summary, key=lambda x: x['avg_cosine_similarity_true'])
best_mse = min(lambda_summary, key=lambda x: x['avg_mse_true'])
best_top2 = max(lambda_summary, key=lambda x: x['avg_top_2_accuracy'])

print(f"Best by cosine similarity: L1={best_cosine['l1_lambda']}, L2={best_cosine['l2_lambda']}, cosine={best_cosine['avg_cosine_similarity_true']:.4f}")
print(f"Best by MSE: L1={best_mse['l1_lambda']}, L2={best_mse['l2_lambda']}, MSE={best_mse['avg_mse_true']:.4f}")
print(f"Best by top-2 accuracy: L1={best_top2['l1_lambda']}, L2={best_top2['l2_lambda']}, acc={best_top2['avg_top_2_accuracy']:.4f}")

# Save detailed results
with open('two_attribute_lambda_grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open('lambda_summary_results.json', 'w') as f:
    json.dump(lambda_summary, f, indent=2)

print(f"\nDetailed results saved to:")
print(f"  - 'two_attribute_lambda_grid_search_results.json' (all results)")
print(f"  - 'lambda_summary_results.json' (averages by lambda combination)")