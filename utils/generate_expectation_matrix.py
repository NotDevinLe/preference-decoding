import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, base_prompt
from drift import get_log_probs
import os
from tqdm import tqdm

def generate_expectation_matrix(model, tokenizer, prompts, num_samples, device):
    """Generate expectation matrix for given prompts."""
    
    # Prepare all prompts at once
    all_inputs = []
    for prompt in prompts:
        formatted_input = tokenizer.apply_chat_template([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        all_inputs.append(formatted_input)
    
    # Generate all outputs at once - vLLM handles batching internally
    print(f"Generating {num_samples} outputs for {len(all_inputs)} prompts...")
    sampling_params = SamplingParams(
        temperature=1.0, 
        max_tokens=1024, 
        n=num_samples
    )
    
    all_outputs = model.generate(all_inputs, sampling_params)
    
    # Extract all generated texts and prepare for reward computation
    print("Preparing data for reward computation...")
    all_reward_data = []
    output_mapping = []  # Track which outputs belong to which prompt
    
    for prompt_idx, vllm_output in enumerate(all_outputs):
        prompt = prompts[prompt_idx]
        for sample_idx, output in enumerate(vllm_output.outputs):
            all_reward_data.append((prompt, output.text))
            output_mapping.append((prompt_idx, sample_idx))
    
    # Compute rewards
    print(f"Computing rewards for {len(all_reward_data)} generated outputs...")
    all_rewards = compute_rewards(model, tokenizer, all_reward_data, device)
    
    # Create expectation matrix
    expectation_matrix = torch.zeros((len(prompts), num_samples, len(attribute_prompts)), device=device)
    
    # Map rewards back to expectation matrix
    for idx, (prompt_idx, sample_idx) in enumerate(output_mapping):
        expectation_matrix[prompt_idx, sample_idx] = all_rewards[idx]
    
    return expectation_matrix

def compute_rewards(model, tokenizer, data, device):
    """Compute reward vectors for data."""
    m = len(data)
    
    # Flatten all data for batch processing
    flat_questions = []
    flat_outputs = []
    
    for prompt, output in data:
        flat_questions.append(prompt)
        flat_outputs.append(output)
    
    # Get base log probabilities for all flattened items
    print("Computing base log probabilities...")
    base_probs, base_counts = get_log_probs(
        model, tokenizer, [base_prompt] * m, 
        flat_questions, flat_outputs, device
    )
    base_tensor = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Initialize drift scores for all items
    drift_scores = torch.zeros((m, len(attribute_prompts)), device=device)
    
    # Process each attribute prompt individually with progress bar
    print(f"Computing attribute log probabilities for {len(attribute_prompts)} attributes...")
    for i, attribute_prompt in enumerate(tqdm(attribute_prompts, desc="Processing attributes")):
        # Get log probabilities for this attribute prompt
        attr_probs, attr_counts = get_log_probs(
            model, tokenizer, [attribute_prompt] * m, 
            flat_questions, flat_outputs, device
        )
        
        # Convert to tensors
        attr_tensor = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Compute drift contribution for this attribute
        attribute_drift = (attr_tensor - base_tensor)
        
        # Set drift scores for this attribute
        drift_scores[:, i] = attribute_drift
    
    return drift_scores

def main():
    parser = argparse.ArgumentParser(description="Generate and save expectation matrix")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to file containing prompts")
    parser.add_argument("--num_expectation_samples", type=int, default=100, help="Number of expectation samples per prompt")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save expectation matrix")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit number of prompts to process")
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_id}")
    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=8192
    )
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load prompts
    print(f"Loading prompts from: {args.prompts_file}")
    with open(args.prompts_file, "r") as f:
        data = json.load(f)
    
    # Extract just the prompts
    if isinstance(data[0], dict) and 'prompt' in data[0]:
        prompts = [item['prompt'] for item in data]
    else:
        prompts = data
    
    # Optionally limit size
    if args.sample_size:
        prompts = prompts[:args.sample_size]
        print(f"Using {args.sample_size} prompts")
    else:
        print(f"Using all {len(prompts)} prompts")
    
    # Generate expectation matrix
    print(f"\nGenerating expectation matrix...")
    print(f"Number of attributes: {len(attribute_prompts)}")
    print(f"Number of expectation samples per prompt: {args.num_expectation_samples}")
    
    expectation_matrix = generate_expectation_matrix(
        model, tokenizer, prompts, args.num_expectation_samples, device
    )
    
    # Save expectation matrix
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"\nSaving expectation matrix to: {args.output_path}")
    torch.save({
        'expectation_matrix': expectation_matrix.cpu(),
        'num_expectation_samples': args.num_expectation_samples,
        'num_attributes': len(attribute_prompts),
        'num_prompts': len(prompts)
    }, args.output_path)
    
    print(f"Expectation matrix saved successfully!")
    print(f"Shape: {expectation_matrix.shape}")

if __name__ == "__main__":
    main()