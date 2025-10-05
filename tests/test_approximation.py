import asyncio
import json
import numpy as np
import argparse
from transformers import AutoTokenizer
import random

# Import async utils for log prob computation and drift approximation
from src.core.drift import approximate, evaluate_accuracy

# Import attribute prompts - support both local import and config file
import sys
import os
sys.path.append('../src')

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

async def main():
    parser = argparse.ArgumentParser(description="Test preference approximation using async VLLM")
    parser.add_argument("--train-data", type=str, default="data/persona_pref/user18_train.json", help="Training data path")
    parser.add_argument("--test-data", type=str, default="data/persona_pref/user18_test.json", help="Test data path")
    parser.add_argument("--l1-lambda", type=float, default=0.01, help="L1 regularization parameter")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8080", help="Gateway URL")
    
    args = parser.parse_args()
    
    with open("configs/attribute_prompts_400.json", "r") as f:
        attribute_prompts = json.load(f)["prompts"]

    selected_attributes =  [366, 202, 229, 230, 282]
    selected_attributes = [attribute_prompts[i] for i in selected_attributes]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_prompt = "You are an AI assistant."
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    print(f"Loading training data from {args.train_data}")
    with open(args.train_data, 'r') as f:
        train_data_raw = json.load(f)
    
    # Convert to drift format: (question, chosen, rejected)
    train_data = []
    for i, item in enumerate(train_data_raw):
        train_data.append((item['prompt'], item['chosen'], item['rejected']))
    
    print(f"Loaded {len(train_data)} training samples")
    
    # Find p vector
    print("Finding p vector...")
    p = await approximate(args.gateway_url, train_data, tokenizer, MODEL_ID, base_prompt, selected_attributes, args.l1_lambda)
    
    print(f"Found p vector with {np.count_nonzero(p)} non-zero components")
    print(f"P vector norm: {np.linalg.norm(p):.4f}")
    print(f"Top 5 attributes by weight:")
    top_indices = np.argsort(np.abs(p))[-5:][::-1]
    for i in top_indices:
        print(f"  {i}: {p[i]:.4f} - {selected_attributes[i][:80]}...")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluate accuracy
    print("\nEvaluating accuracy on test data...")
    accuracy = await evaluate_accuracy(args.gateway_url, test_data, p, tokenizer, MODEL_ID, base_prompt, selected_attributes)

    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    asyncio.run(main())