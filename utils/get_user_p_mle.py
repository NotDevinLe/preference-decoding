import argparse
import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
from mle import MLE
from attribute_prompts import attribute_prompts
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Train MLE and get p vector from expectation matrix")
    parser.add_argument("--name", type=str, default="user1", help="User name for data files")
    parser.add_argument("--num_expectation_samples", type=int, default=16, help="Number of expectation samples per prompt")
    parser.add_argument("--sample_size", type=int, default=200, help="Training data size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--num_mc_samples", type=int, default=10, help="Number of MC samples for expectation")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for tracking")
    parser.add_argument("--wandb_project", type=str, default="mle-preference", help="Wandb project name")
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
        return
    
    # Limit data size
    if args.sample_size:
        train_data = train_data[:args.sample_size]
        print(f"Using {args.sample_size} training samples")
    
    # Load expectation matrix
    expectation_matrix_path = f"../data/expectation_matrices/{args.name}_expectation_n{args.num_expectation_samples}_size{args.sample_size}.pt"
    
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
    
    print(" Expectation matrix loaded successfully and is compatible")
    print(f"  Expectation matrix: {checkpoint['expectation_matrix'].shape}")
    print(f"  Chosen rewards: {checkpoint['chosen_rewards'].shape}")
    
    # Initialize MLE with loaded matrices
    print("\\nInitializing MLE model...")
    mle_model = MLE(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        device=device,
        expectation_matrix=checkpoint['expectation_matrix'],
        chosen_rewards=checkpoint['chosen_rewards'],
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )

    mle_model.p = torch.tensor([
    8.260398864746094,
    -0.6596850156784058,
    1.1252026557922363,
    -0.2302657663822174,
    3.111055612564087,
    -2.367696762084961,
    -1.0958409309387207,
    -1.5982941389083862,
    3.1958601474761963,
    -3.3132991790771484,
    2.665332317352295,
    1.503035545349121,
    -0.016793513670563698,
    3.105441093444824,
    0.662980318069458,
    0.39985281229019165,
    -0.47467970848083496,
    2.073554754257202,
    2.6395182609558105,
    0.6215099692344666,
    2.196147918701172,
    8.028526306152344,
    -1.1944458484649658,
    -1.708486557006836,
    3.085697650909424,
    2.577425479888916
  ], device=device)
    
    # Train MLE to learn p vector
    print("\\nStarting MLE training...")
    print(f"Training parameters:")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Beta: {args.beta}")
    print(f"  MC samples: {args.num_mc_samples}")
    
    mle_model.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_mc_samples=args.num_mc_samples
    )
    
    # Save results
    results_dir = "../results/mle"
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = f"{results_dir}/{args.name}_p_vector_n{args.num_expectation_samples}_size{args.sample_size}.json"
    print(f"\\nSaving p vector to: {save_path}")
    mle_model.save_results(save_path)
    
    # Print final p vector
    final_p = mle_model.p.cpu().numpy()
    print("\\n" + "="*60)
    print("FINAL P VECTOR RESULTS")
    print("="*60)
    print(f"P vector norm: {np.linalg.norm(final_p):.4f}")
    print(f"P vector: {final_p}")
    
    # Print top attributes by absolute weight
    print("\\nTop 10 attributes by absolute weight:")
    top_indices = np.argsort(np.abs(final_p))[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. Attribute {idx:2d}: {final_p[idx]:+7.4f}")
        print(f"      Prompt: {attribute_prompts[idx][:80]}...")
    
    # Print summary statistics
    print(f"\\nSummary Statistics:")
    print(f"  Positive weights: {np.sum(final_p > 0)}")
    print(f"  Negative weights: {np.sum(final_p < 0)}")
    print(f"  Zero weights: {np.sum(np.abs(final_p) < 1e-6)}")
    print(f"  Max weight: {np.max(final_p):.4f}")
    print(f"  Min weight: {np.min(final_p):.4f}")
    print(f"  Mean absolute weight: {np.mean(np.abs(final_p)):.4f}")
    
    print("\\n MLE training complete! P vector saved.")

if __name__ == "__main__":
    main()