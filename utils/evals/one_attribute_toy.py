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

def generate_data(system_prompt, base_prompt, size, dolly_ds):
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

    attr_prompt_inputs = []
    attr_prompt_outputs = []

    for instruction in instructions:
        attr_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        attr_prompt_inputs.append(attr_prompt_input)

    attr_prompt_outputs = llm.generate(attr_prompt_inputs, sampling_params)
    attr_prompt_outputs = [output.outputs[0].text.strip() for output in attr_prompt_outputs]

    all_data = []
    for i in range(len(instructions)):
        if i < len(attr_prompt_outputs):
            # Use attribute prompt output for chosen
            all_data.append({
                "prompt": instructions[i],
                "chosen": attr_prompt_outputs[i],
                "rejected": base_prompt_outputs[i]
            })
        else:
            # Use base prompt output for both (no preference)
            all_data.append({
                "prompt": instructions[i],
                "chosen": base_prompt_outputs[i],
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

# Run experiment for each attribute
results = []
train_size, test_size = 200, 1000

for experiment_idx, attr_prompt in enumerate(selected_prompts):
    print(f"\n=== EXPERIMENT {experiment_idx + 1}: {attr_prompt[:50]}... ===")
    
    # Generate training data
    train_data = generate_data(attr_prompt, base_prompt, train_size, dolly_ds)
    
    # Generate test data (for evaluation)
    test_data = generate_data(attr_prompt, base_prompt, test_size, dolly_ds)
    
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

    # Check which attribute should be active
    print(f"Expected active attribute: {experiment_idx}")
    means = torch.mean(training_matrix, dim=0).cpu().numpy()
    print(f"Highest signal attribute: {np.argmax(np.abs(means))} (value={means[np.argmax(np.abs(means))]:.6f})")
        
    # Compute average preference vector
    p_recovered = torch.mean(training_matrix, dim=0).cpu().numpy()
    
    # Normalize
    if np.linalg.norm(p_recovered, ord=1) > 1:
        p_recovered = p_recovered * (1 / np.linalg.norm(p_recovered, ord=1))
    
    # True distribution (ground truth)
    true_p = np.zeros(len(selected_prompts))
    true_p[experiment_idx] = 1.0
    
    # Calculate recovery metrics
    mse = np.mean((p_recovered - true_p) ** 2)
    mae = np.mean(np.abs(p_recovered - true_p))
    cosine_sim = np.dot(p_recovered, true_p) / (np.linalg.norm(p_recovered) * np.linalg.norm(true_p) + 1e-8)
    
    # Check if correct attribute is identified as top
    top_recovered_idx = np.argmax(p_recovered)
    correct_top_1 = 1.0 if top_recovered_idx == experiment_idx else 0.0
    
    result = {
        'experiment': experiment_idx + 1,
        'attribute': attr_prompt,
        'true_distribution': true_p.tolist(),
        'recovered_distribution': p_recovered.tolist(),
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cosine_sim,
        'top_1_accuracy': correct_top_1,
    }
    
    results.append(result)
    
    print(f"\nResults for Experiment {experiment_idx + 1}:")
    print(f"  True distribution: {true_p}")
    print(f"  Recovered distribution: {p_recovered}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Top-1 accuracy: {correct_top_1:.1f}/1")
    
# Summary statistics
print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL 5 EXPERIMENTS")
print(f"{'='*60}")

avg_mse = np.mean([r['mse'] for r in results])
avg_mae = np.mean([r['mae'] for r in results])
avg_cosine = np.mean([r['cosine_similarity'] for r in results])
avg_top1_acc = np.mean([r['top_1_accuracy'] for r in results])

print(f"Average MSE: {avg_mse:.4f} ± {np.std([r['mse'] for r in results]):.4f}")
print(f"Average MAE: {avg_mae:.4f} ± {np.std([r['mae'] for r in results]):.4f}")
print(f"Average Cosine Similarity: {avg_cosine:.4f} ± {np.std([r['cosine_similarity'] for r in results]):.4f}")
print(f"Average Top-1 Accuracy: {avg_top1_acc:.2f} ± {np.std([r['top_1_accuracy'] for r in results]):.2f}")

# Save detailed results
with open('one_attribute_toy_results_temperature_0.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to 'one_attribute_toy_results_temperature_0.json'")
