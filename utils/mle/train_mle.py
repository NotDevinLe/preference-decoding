import argparse
import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
import os
import numpy as np
import sys

# Add parent directory to path to import mle module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mle import MLE
from attribute_prompts import attribute_prompts, base_prompt

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
    parser.add_argument("--load_chosen_rewards", action="store_true", help="Load pre-computed chosen rewards if available")
    # Convergence criteria
    parser.add_argument("--max_epochs", type=int, default=10000, help="Maximum number of epochs")
    parser.add_argument("--gradient_tolerance", type=float, default=1e-6, help="Stop when gradient norm is below this threshold")
    parser.add_argument("--loss_tolerance", type=float, default=1e-6, help="Stop when loss change is below this threshold")
    parser.add_argument("--patience", type=int, default=100, help="Stop if no improvement for this many epochs")
    parser.add_argument("--l1_lambda", type=float, default=0.0, help="L1 regularization coefficient (0.0 = no regularization)")
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
    train_data_path = f"../../data/preference/{args.name}_train.json"
    print(f"Loading training data from: {train_data_path}")
    
    try:
        with open(train_data_path, "r") as f:
            full_train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find training data at {train_data_path}")
        print("Make sure you have generated the preference data first.")
        return
    
    # Store full data size before slicing
    full_data_size = len(full_train_data)
    
    # Optionally limit data size for testing
    if args.sample_size:
        train_data = full_train_data[:args.sample_size]
        print(f"Using {args.sample_size} training samples (out of {full_data_size} total)")
    else:
        train_data = full_train_data
        print(f"Using all {len(train_data)} training samples")
    
    # Always load the n=200 size=200 expectation matrix and slice as needed
    expectation_matrix_path = f"../../data/expectation_matrices/{args.name}_expectation_n200_size200.pt"
    
    if not os.path.exists(expectation_matrix_path):
        print(f"Error: Expectation matrix not found at {expectation_matrix_path}")
        print("Please run generate_expectation_matrix.py first")
        return
    
    print(f"Loading expectation matrix from: {expectation_matrix_path}")
    checkpoint = MLE.load_expectation_matrix(expectation_matrix_path)
    loaded_matrix = checkpoint['expectation_matrix']
    
    # Slice expectation matrix to match both dataset size and num_expectation_samples
    # loaded_matrix shape is (200, 200, num_attributes)
    matrix_to_use = loaded_matrix
    
    # First slice by datapoints
    num_datapoints = len(train_data)
    if num_datapoints < loaded_matrix.shape[0]:
        print(f"Slicing expectation matrix datapoints from {loaded_matrix.shape[0]} to {num_datapoints}")
        matrix_to_use = matrix_to_use[:num_datapoints]
    elif num_datapoints > loaded_matrix.shape[0]:
        print(f"Error: Dataset size {num_datapoints} is larger than available expectation matrix size {loaded_matrix.shape[0]}")
        return
    
    # Then slice by expectation samples
    if args.num_expectation_samples < loaded_matrix.shape[1]:
        print(f"Slicing expectation samples from {loaded_matrix.shape[1]} to {args.num_expectation_samples}")
        matrix_to_use = matrix_to_use[:, :args.num_expectation_samples, :]
    elif args.num_expectation_samples > loaded_matrix.shape[1]:
        print(f"Error: Requested num_expectation_samples {args.num_expectation_samples} is larger than available {loaded_matrix.shape[1]}")
        return
    
    checkpoint['expectation_matrix'] = matrix_to_use
    
    # Verify compatibility after potential slicing
    expected_shape = (len(train_data), args.num_expectation_samples, len(attribute_prompts))
    if checkpoint['expectation_matrix'].shape != expected_shape:
        print(f"Error: Expectation matrix shape {checkpoint['expectation_matrix'].shape} doesn't match expected {expected_shape}")
        return
    
    # Optionally load chosen rewards
    chosen_rewards = None
    if args.load_chosen_rewards:
        # Always load from the size 200 chosen rewards file
        chosen_rewards_path = f"../../data/chosen_rewards/{args.name}_chosen_rewards.pt"
        if os.path.exists(chosen_rewards_path):
            print(f"Loading chosen rewards from: {chosen_rewards_path}")
            rewards_checkpoint = MLE.load_chosen_rewards(chosen_rewards_path)
            loaded_rewards = rewards_checkpoint['chosen_rewards']
            
            # Slice chosen rewards to match dataset size if using sample_size
            if args.sample_size and args.sample_size < loaded_rewards.shape[0]:
                print(f"Slicing chosen rewards from {loaded_rewards.shape[0]} to {args.sample_size} datapoints")
                chosen_rewards = loaded_rewards[:args.sample_size]
            else:
                chosen_rewards = loaded_rewards
        else:
            print(f"Chosen rewards file not found at {chosen_rewards_path}, will compute from scratch")
    
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
        chosen_rewards=chosen_rewards,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Train MLE
    print("\nStarting MLE training...")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    print(f"MC samples: {args.num_mc_samples}")
    print(f"Gradient tolerance: {args.gradient_tolerance}")
    print(f"Loss tolerance: {args.loss_tolerance}")
    print(f"Patience: {args.patience}")
    print(f"L1 regularization: {args.l1_lambda}")
    
    mle_model.train(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_mc_samples=args.num_mc_samples,
        gradient_tolerance=args.gradient_tolerance,
        loss_tolerance=args.loss_tolerance,
        patience=args.patience,
        l1_lambda=args.l1_lambda
    )
    
    # Save results
    results_dir = "../../results/mle"
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = f"{results_dir}/{args.name}_lambda.jsonl"
    print(f"\nSaving results to: {save_path}")
    mle_model.save_results(save_path, args.num_mc_samples, args.l1_lambda)
    
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