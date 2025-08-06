#!/usr/bin/env python3
"""
Example usage of the updated MLE class with flexible expectation matrix generation.
"""

import json
import torch
from vllm import LLM
from transformers import AutoTokenizer
from utils.mle import MLE

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Load model and tokenizer
    model = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    with open("data/preference/user1_train.json", "r") as f:
        train_data = json.load(f)[:10]  # Use subset for example
    
    print("=== Method 1: Generate expectation matrix from scratch ===")
    # Initialize MLE without expectation matrix
    mle_model = MLE(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        device=device,
        num_expectation_samples=16,  # Default: 100, using 16 for speed
        use_wandb=False
    )
    
    # Generate expectation matrix for this user's data
    mle_model.generate_expectation_matrix()
    
    # Now train
    mle_model.train(num_epochs=10, learning_rate=0.01)
    
    print("\n=== Method 2: Load pre-computed expectation matrix ===")
    # Load pre-computed expectation matrix
    checkpoint = MLE.load_expectation_matrix("data/expectation_matrices/user1_expectation_n16_size200.pt")
    
    # Initialize MLE with pre-computed matrix
    mle_model2 = MLE(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        device=device,
        expectation_matrix=checkpoint['expectation_matrix'][:10],  # Use subset
        use_wandb=False
    )
    
    # Train directly (no need to generate expectation matrix)
    mle_model2.train(num_epochs=10, learning_rate=0.01)
    
    print("\n=== Method 3: What happens if you try to train without generating matrix ===")
    # Initialize MLE without expectation matrix
    mle_model3 = MLE(
        model=model,
        tokenizer=tokenizer,
        data=train_data,
        device=device,
        use_wandb=False
    )
    
    # Try to train without generating matrix (will get warning and auto-generate)
    mle_model3.train(num_epochs=5, learning_rate=0.01)

if __name__ == "__main__":
    main()