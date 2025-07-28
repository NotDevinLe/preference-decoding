import numpy as np
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
import gc
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.drift import get_training_matrix
    from utils.attribute_prompts import base_prompt
except ImportError:
    # If running from evals folder directly
    sys.path.append('..')
    from drift import get_training_matrix
    from attribute_prompts import base_prompt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def generate_data_batched(system_prompt, base_prompt, size, dolly_ds, batch_size=50):
    """Generate preference data in batches to save memory"""
    print(f"Generating data in batches of {batch_size}...")
    
    instructions = [build_prompt(row["instruction"], row["context"]) 
                   for row in dolly_ds.shuffle().select(range(size))]
    
    all_preference_data = []
    
    # Process in batches
    for batch_start in range(0, size, batch_size):
        batch_end = min(batch_start + batch_size, size)
        batch_instructions = instructions[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(size-1)//batch_size + 1} "
              f"(items {batch_start}-{batch_end-1})")
        
        # Generate base prompt responses for this batch
        base_prompt_inputs = []
        for instruction in batch_instructions:
            base_prompt_input = tokenizer.apply_chat_template([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": instruction}
            ], tokenize=False, add_generation_prompt=True)
            base_prompt_inputs.append(base_prompt_input)

        base_prompt_outputs = llm.generate(base_prompt_inputs, sampling_params)
        base_prompt_outputs = [output.outputs[0].text.strip() for output in base_prompt_outputs]

        # Generate attribute prompt responses for this batch
        attr_prompt_inputs = []
        for instruction in batch_instructions:
            attr_prompt_input = tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ], tokenize=False, add_generation_prompt=True)
            attr_prompt_inputs.append(attr_prompt_input)

        attr_prompt_outputs = llm.generate(attr_prompt_inputs, sampling_params)
        attr_prompt_outputs = [output.outputs[0].text.strip() for output in attr_prompt_outputs]

        # Create preference pairs for this batch
        for i in range(len(batch_instructions)):
            all_preference_data.append({
                "prompt": batch_instructions[i],
                "chosen": attr_prompt_outputs[i],
                "rejected": base_prompt_outputs[i]
            })
        print(all_preference_data)
        # Clear memory after each batch
        clear_memory()

    return all_preference_data

def build_prompt(instruction, context):
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction

# Simple 2-attribute test: English vs Spanish
selected_prompts = [
    "You are an AI assistant that only responds in English.",
    "Eres un asistente de IA que responde únicamente en español. Todas tus respuestas deben estar en español.",
]

print("=== MEMORY-OPTIMIZED ENGLISH VS SPANISH TEST ===")
print(f"Testing with {len(selected_prompts)} attributes:")
for i, prompt in enumerate(selected_prompts):
    print(f"  {i}: {prompt}")

# Check GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory available: {gpu_memory:.1f} GB")

# Model setup - more conservative settings for Titan GPU
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# More conservative vLLM settings
llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,  # Reduced from 0.7
    max_model_len=2048,  # Reduced from 4096
    swap_space=2,  # Add swap space
    cpu_offload_gb=2,  # Offload some weights to CPU
)

# More conservative sampling - shorter responses
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=256,  # Reduced from 512
    stop=[]
)

# Load Dolly dataset
dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# Reduced experiment parameters
train_size = 5  # Reduced from 200
batch_size = 25   # Process data in smaller batches
results = []

print(f"Using reduced train_size: {train_size} (from 200)")
print(f"Using batch processing: {batch_size} items per batch")

for experiment_idx, attr_prompt in enumerate(selected_prompts):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_idx + 1}: {attr_prompt}")
    print(f"{'='*60}")
    
    # Clear memory before each experiment
    clear_memory()
    
    # Generate training data where this attribute is preferred
    print(f"Generating {train_size} preference pairs in batches...")
    train_data = generate_data_batched(attr_prompt, base_prompt, train_size, dolly_ds, batch_size)
    
    print(f"Sample preference pair:")
    print(f"  Prompt: {train_data[0]['prompt'][:100]}...")
    print(f"  Chosen: {train_data[0]['chosen'][:100]}...")
    print(f"  Rejected: {train_data[0]['rejected'][:100]}...")
    
    # Clear memory before drift computation
    clear_memory()
    
    # Run drift approximation to recover preference vector
    print(f"\nRunning drift approximation...")
    try:
        training_matrix = get_training_matrix(
            [(item['prompt'], item['chosen'], item['rejected']) for item in train_data], 
            llm, tokenizer, base_prompt, selected_prompts, device
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM during drift computation. Trying with smaller batch or clearing cache...")
            clear_memory()
            # Retry with half the data if OOM
            half_data = train_data[:len(train_data)//2]
            print(f"Retrying with {len(half_data)} samples instead of {len(train_data)}")
            training_matrix = get_training_matrix(
                [(item['prompt'], item['chosen'], item['rejected']) for item in half_data], 
                llm, tokenizer, base_prompt, selected_prompts, device
            )
        else:
            raise e

    print(f"\nTraining matrix analysis:")
    print(f"  Shape: {training_matrix.shape}")
    print(f"  Range: [{training_matrix.min():.6f}, {training_matrix.max():.6f}]")
    print(f"  Mean per attribute: {torch.mean(training_matrix, dim=0).cpu().numpy()}")
    print(f"  Std per attribute: {torch.std(training_matrix, dim=0).cpu().numpy()}")
    
    # Compute recovered preference vector
    p_recovered = torch.mean(training_matrix, dim=0).cpu().numpy()
    
    # Clear memory after computation
    clear_memory()

    print("training matrix")
    print(training_matrix)
    
    # Normalize to unit L1 norm if needed
    l1_norm = np.linalg.norm(p_recovered, ord=1)
    if l1_norm > 1:
        p_recovered = p_recovered / l1_norm
    
    # Ground truth: one-hot vector for current attribute
    true_p = np.zeros(len(selected_prompts))
    true_p[experiment_idx] = 1.0
    
    # Calculate metrics
    mse = float(np.mean((p_recovered - true_p) ** 2))
    mae = float(np.mean(np.abs(p_recovered - true_p)))
    
    # Cosine similarity
    cos_sim = float(np.dot(p_recovered, true_p) / (np.linalg.norm(p_recovered) * np.linalg.norm(true_p) + 1e-8))
    
    # Top-1 accuracy: did we identify the correct attribute as most important?
    top_recovered_idx = int(np.argmax(np.abs(p_recovered)))
    correct_top_1 = 1.0 if top_recovered_idx == experiment_idx else 0.0
    
    # Store results with explicit type conversion
    result = {
        'experiment': int(experiment_idx + 1),
        'attribute': attr_prompt,
        'expected_attribute_idx': int(experiment_idx),
        'recovered_attribute_idx': int(top_recovered_idx),
        'true_distribution': [float(x) for x in true_p.tolist()],
        'recovered_distribution': [float(x) for x in p_recovered.tolist()],
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cos_sim,
        'top_1_accuracy': correct_top_1,
        'l1_norm_before_normalization': float(l1_norm)
    }
    
    results.append(result)
    
    # Print results
    print(f"\nRESULTS:")
    print(f"  Expected (ground truth): {true_p}")
    print(f"  Recovered: {p_recovered}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Top-1 accuracy: {correct_top_1}")
    print(f"  Expected attribute: {experiment_idx} ({attr_prompt})")
    print(f"  Recovered attribute: {top_recovered_idx} ({selected_prompts[top_recovered_idx]})")
    
    if correct_top_1 == 1.0:
        print(f"  ✅ SUCCESS: Correctly identified the active attribute!")
    else:
        print(f"  ❌ FAILURE: Incorrectly identified attribute {top_recovered_idx} instead of {experiment_idx}")

# Summary across both experiments
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

avg_mse = np.mean([r['mse'] for r in results])
avg_mae = np.mean([r['mae'] for r in results])
avg_cosine = np.mean([r['cosine_similarity'] for r in results])
avg_top1_acc = np.mean([r['top_1_accuracy'] for r in results])

print(f"Average MSE: {avg_mse:.6f}")
print(f"Average MAE: {avg_mae:.6f}")
print(f"Average Cosine Similarity: {avg_cosine:.6f}")
print(f"Average Top-1 Accuracy: {avg_top1_acc:.2f} ({int(avg_top1_acc * len(results))}/{len(results)} correct)")

success_rate = avg_top1_acc
if success_rate == 1.0:
    print(f"\n🎉 PERFECT SUCCESS! The algorithm correctly identified the active attribute in all experiments.")
elif success_rate >= 0.5:
    print(f"\n✅ GOOD PERFORMANCE! Success rate: {success_rate:.1%}")
else:
    print(f"\n⚠️  POOR PERFORMANCE! Success rate: {success_rate:.1%}")

# Save results
output_file = 'english_spanish_recovery_test_optimized.json'
with open(output_file, 'w') as f:
    json.dump({
        'experiment_config': {
            'attributes': selected_prompts,
            'train_size': train_size,
            'batch_size': batch_size,
            'model': model_id,
            'temperature': 0.0,
            'max_tokens': 256,
            'max_model_len': 2048
        },
        'results': results,
        'summary': {
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            'avg_cosine_similarity': avg_cosine,
            'avg_top1_accuracy': avg_top1_acc
        }
    }, f, indent=2)

print(f"\nDetailed results saved to '{output_file}'")

# Final memory cleanup
clear_memory()
print("Memory cleanup completed.")