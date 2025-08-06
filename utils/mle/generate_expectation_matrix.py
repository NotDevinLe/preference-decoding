import argparse
import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
from mle import MLE
import os


def main():
    parser = argparse.ArgumentParser(description="Generate and save expectation matrix")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to preference training data file (JSON with prompt/chosen/rejected format)")
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
    
    # Load training data (preference format)
    print(f"Loading training data from: {args.prompts_file}")
    with open(args.prompts_file, "r") as f:
        data = json.load(f)
    
    # Optionally limit size
    if args.sample_size:
        data = data[:args.sample_size]
        print(f"Using {args.sample_size} data points")
    else:
        print(f"Using all {len(data)} data points")
    
    print(f"\nGenerating expectation matrix using MLE class...")
    print(f"Number of expectation samples per prompt: {args.num_expectation_samples}")
    
    # Initialize MLE without expectation matrix (will generate from scratch)
    mle_model = MLE(
        model=model,
        tokenizer=tokenizer,
        data=data,
        device=device,
        num_expectation_samples=args.num_expectation_samples,
        use_wandb=False  # Don't use wandb for standalone generation
    )
    
    # Generate expectation matrix
    mle_model.generate_expectation_matrix()
    
    # Save expectation matrix using MLE's method
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    mle_model.save_expectation_matrix(args.output_path)
    
    print(f"\nExpectation matrix generation complete!")
    print(f"Saved to: {args.output_path}")
    print(f"Shape: {mle_model.expectation.shape}")

if __name__ == "__main__":
    main()