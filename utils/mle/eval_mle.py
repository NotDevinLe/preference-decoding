import argparse
import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
from mle import MLE
from attribute_prompts import attribute_prompts, base_prompt
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLE for preference learning")
    parser.add_argument("--name", type=str, default="user1", help="User name for data files")
    parser.add_argument("--num_expectation_samples", type=int, default=50, help="Number of expectation samples per prompt")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--num_mc_samples", type=int, default=10, help="Number of MC samples for expectation")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for tracking")
    parser.add_argument("--wandb_project", type=str, default="mle-preference", help="Wandb project name")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit training data size (for testing)")
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
        max_model_len=4096
    )
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load training data
    train_data_path = f"../data/preference/{args.name}_train.json"
    print(f"Loading training data from: {train_data_path}")
    
    try:
        with open(train_data_path, "r") as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find training data at {train_data_path}")
        print("Make sure you have generated the preference data first.")
        return
    
    # Optionally limit data size for testing
    if args.sample_size:
        train_data = train_data[:args.sample_size]
        print(f"Using {args.sample_size} training samples")
    else:
        print(f"Using all {len(train_data)} training samples")
    
    # Load expectation matrix
    expectation_matrix_path = f"../data/expectation_matrices/{args.name}_expectation_n{args.num_expectation_samples}_size{len(train_data)}.pt"
    
    if not os.path.exists(expectation_matrix_path):
        print(f"Error: Expectation matrix not found at {expectation_matrix_path}")
        print("Please run generate_expectation_matrix.py first")
        return
    
    print(f"Loading expectation matrix from: {expectation_matrix_path}")
    checkpoint = MLE.load_expectation_matrix(expectation_matrix_path)
    
    # Verify compatibility
    expected_shape = (len(train_data), args.num_expectation_samples, len(attribute_prompts))
    if checkpoint['expectation_matrix'].shape != expected_shape:
        print(f"Error: Expectation matrix shape {checkpoint['expectation_matrix'].shape} doesn't match expected {expected_shape}")
        return
    
    # Initialize MLE
    print("\nInitializing MLE model...")
    print(f"Number of attributes: {len(attribute_prompts)}")
    print(f"Number of expectation samples per prompt: {args.num_expectation_samples}")
    print(f"Expectation matrix loaded successfully and is compatible")
    print(f"  Expectation matrix: {checkpoint['expectation_matrix'].shape}")
    
    mle_model = MLE(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        device=device,
        expectation_matrix=checkpoint['expectation_matrix'],
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Train MLE
    print("\nStarting MLE training...")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    print(f"MC samples: {args.num_mc_samples}")
    
    mle_model.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_mc_samples=args.num_mc_samples
    )
    
    # Save results
    results_dir = "../results/mle"
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = f"{results_dir}/{args.name}_p_vector.json"
    print(f"\nSaving results to: {save_path}")
    mle_model.save_results(save_path)
    
    # Print final p vector
    final_p = mle_model.p.cpu().numpy()
    print("\nFinal p vector:")
    for i, (attr, p_val) in enumerate(zip(attribute_prompts, final_p)):
        print(f"  Attribute {i}: {p_val:.4f}")
        print(f"    Prompt: {attr[:50]}...")
    
    # Print top attributes by weight
    print("\nTop 5 attributes by absolute weight:")
    top_indices = np.argsort(np.abs(final_p))[::-1][:5]
    for idx in top_indices:
        print(f"  Attribute {idx}: {final_p[idx]:.4f}")
        print(f"    Prompt: {attribute_prompts[idx][:100]}...")
    
    print("\nMLE evaluation complete!")

if __name__ == "__main__":
    main()