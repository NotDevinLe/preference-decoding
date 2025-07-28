import numpy as np
import random
import itertools
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
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

# Run experiment 10 times
results = []
train_size, test_size = 200, 1000

pairs = list(itertools.combinations(range(5), 2))

for experiment_idx, pair in enumerate(pairs):
    print(f"\n=== EXPERIMENT {pair} ===")
    
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
    
    # Generate test data (for evaluation)
    test_data = generate_data(attr1, attr2, base_prompt, prob1, prob2, test_size, dolly_ds)
    
    print(f"Generated {len(train_data)} training samples")
    
    # Run training algorithm to recover distribution
    training_matrix = get_training_matrix(
        [(item['prompt'], item['chosen'], item['rejected']) for item in train_data], 
        llm, tokenizer, base_prompt, selected_prompts, device
    )

    print(f"\n=== DEBUG INFO ===")
    print(f"Training matrix shape: {training_matrix.shape}")
    print(f"Training matrix range: [{training_matrix.min():.6f}, {training_matrix.max():.6f}]")
    print(f"Training matrix mean per attribute: {torch.mean(training_matrix, dim=0)}")
    print(f"Training matrix std per attribute: {torch.std(training_matrix, dim=0)}")

    # Check which attributes should be active
    print(f"Expected active attributes: {pair[0]} (prob={prob1:.3f}), {pair[1]} (prob={prob2:.3f})")
    means = torch.mean(training_matrix, dim=0).cpu().numpy()
    print(f"Highest signal attribute: {np.argmax(np.abs(means))} (value={means[np.argmax(np.abs(means))]:.6f})")
        
    # Compute average preference vector
    p_recovered = torch.mean(training_matrix, dim=0).cpu().numpy()
    
    # Normalize
    if np.linalg.norm(p_recovered, ord=1) > 1:
        p_recovered = p_recovered * (1 / np.linalg.norm(p_recovered, ord=1))
    
    # True distribution (ground truth)
    true_p = np.zeros(len(selected_prompts))
    attr1_idx = selected_prompts.index(attr1)
    attr2_idx = selected_prompts.index(attr2)
    true_p[attr1_idx] = prob1
    true_p[attr2_idx] = prob2
    
    # Calculate recovery metrics
    mse = np.mean((p_recovered - true_p) ** 2)
    mae = np.mean(np.abs(p_recovered - true_p))
    cosine_sim = np.dot(p_recovered, true_p) / (np.linalg.norm(p_recovered) * np.linalg.norm(true_p) + 1e-8)
    
    # Check if top attributes are correctly identified
    top_2_recovered = np.argsort(p_recovered)[-2:]
    top_2_true = np.argsort(true_p)[-2:]
    correct_top_2 = len(set(top_2_recovered) & set(top_2_true)) / 2
    
    result = {
        'experiment': experiment_idx + 1,
        'attr1': attr1,
        'attr2': attr2,
        'true_prob1': prob1,
        'true_prob2': prob2,
        'true_distribution': true_p.tolist(),
        'recovered_distribution': p_recovered.tolist(),
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cosine_sim,
        'top_2_accuracy': correct_top_2
    }
    
    results.append(result)
    
    print(f"\nResults for Experiment {experiment_idx + 1}:")
    print(f"  True distribution: {true_p}")
    print(f"  Recovered distribution: {p_recovered}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Top-2 accuracy: {correct_top_2:.1f}/2")
    
    # Attribute identification
    top_recovered_attr = selected_prompts[np.argmax(p_recovered)]
    print(f"  Top recovered attribute: {top_recovered_attr[:50]}...")
    print(f"  Expected: {attr1[:50]}... or {attr2[:50]}...")

# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL 10 EXPERIMENTS")
print(f"{'='*60}")

avg_mse = np.mean([r['mse'] for r in results])
avg_mae = np.mean([r['mae'] for r in results])
avg_cosine = np.mean([r['cosine_similarity'] for r in results])
avg_top2_acc = np.mean([r['top_2_accuracy'] for r in results])

print(f"Average MSE: {avg_mse:.4f} ± {np.std([r['mse'] for r in results]):.4f}")
print(f"Average MAE: {avg_mae:.4f} ± {np.std([r['mae'] for r in results]):.4f}")
print(f"Average Cosine Similarity: {avg_cosine:.4f} ± {np.std([r['cosine_similarity'] for r in results]):.4f}")
print(f"Average Top-2 Accuracy: {avg_top2_acc:.2f} ± {np.std([r['top_2_accuracy'] for r in results]):.2f}")

# Save detailed results
with open('two_attribute_toy_results_temperature_0.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to 'two_attribute_toy_results_temperature_0.json'")
