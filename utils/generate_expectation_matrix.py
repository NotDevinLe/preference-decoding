import argparse
import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
from mle import MLE
from attribute_prompts import attribute_prompts
import os

def main():
    parser = argparse.ArgumentParser(description="Generate and save expectation matrix for MLE")
    parser.add_argument("--name", type=str, default="user1", help="User name for data files")
    parser.add_argument("--num_expectation_samples", type=int, default=100, help="Number of expectation samples per prompt")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit training data size")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save expectation matrix")
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
    
    # Load training data
    train_data_path = f"../data/preference/{args.name}_train.json"
    print(f"Loading training data from: {train_data_path}")
    
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    
    # Optionally limit data size
    if args.sample_size:
        train_data = train_data[:args.sample_size]
        print(f"Using {args.sample_size} training samples")
    else:
        print(f"Using all {len(train_data)} training samples")
    
    # Initialize MLE just to generate expectation matrix
    print(f"\nGenerating expectation matrix...")
    print(f"Number of attributes: {len(attribute_prompts)}")
    print(f"Number of expectation samples per prompt: {args.num_expectation_samples}")
    
    mle_model = MLE(
        model=model,
        tokenizer=tokenizer,
        num_expectation_samples=args.num_expectation_samples,
        data=train_data,
        device=device,
        use_wandb=False  # No need for wandb when just generating matrix
    )
    
    # Save expectation matrix
    expectation_dir = "../data/expectation_matrices"
    os.makedirs(expectation_dir, exist_ok=True)
    
    if args.output_path:
        save_path = args.output_path
    else:
        save_path = f"{expectation_dir}/{args.name}_expectation_n{args.num_expectation_samples}"
        if args.sample_size:
            save_path += f"_size{args.sample_size}"
        save_path += ".pt"
    
    print(f"\nSaving expectation matrix to: {save_path}")
    mle_model.save_expectation_matrix(save_path)
    
    print("\nExpectation matrix generation complete!")

if __name__ == "__main__":
    main()