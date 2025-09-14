#!/usr/bin/env python3
"""
Test Approximation Script
Uses async utils to compute log probs, finds p vector using drift approximation logic,
then evaluates preference pair accuracy on test data.
"""

import asyncio
import json
import numpy as np
import argparse
from transformers import AutoTokenizer

# Import async utils for log prob computation and drift approximation
from async_utils import approximate_async, evaluate_accuracy_async, MODEL_ID

# Import attribute prompts
import sys
sys.path.append('../utils')
from attribute_prompts import attribute_prompts, base_prompt


async def main():
    parser = argparse.ArgumentParser(description="Test preference approximation using async VLLM")
    parser.add_argument("--train-data", type=str, default="data/persona_pref/user11_train.json", help="Training data path")
    parser.add_argument("--test-data", type=str, default="data/persona_pref/user11_test.json", help="Test data path")
    parser.add_argument("--max-train-samples", type=int, default=150, help="Max training samples")
    parser.add_argument("--max-attributes", type=int, default=50, help="Max attribute prompts to use")
    parser.add_argument("--l1-lambda", type=float, default=0.01, help="L1 regularization parameter")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    print(f"Loading training data from {args.train_data}")
    with open(args.train_data, 'r') as f:
        train_data_raw = json.load(f)
    
    # Convert to drift format: (question, chosen, rejected)
    train_data = []
    for i, item in enumerate(train_data_raw[:args.max_train_samples]):
        train_data.append((item['prompt'], item['chosen'], item['rejected']))
    
    print(f"Loaded {len(train_data)} training samples")
    
    # Use subset of attribute prompts
    selected_attributes = attribute_prompts[:args.max_attributes]
    print(f"Using {len(selected_attributes)} attribute prompts")
    
    # Find p vector
    print("Finding p vector...")
    p = await approximate_async(train_data, tokenizer, base_prompt, selected_attributes, args.l1_lambda)
    
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
    accuracy = await evaluate_accuracy_async(test_data, p, tokenizer, base_prompt, selected_attributes)
    
    print(f"\nResults:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Attribute prompts: {len(selected_attributes)}")
    print(f"Non-zero p components: {np.count_nonzero(p)}")
    print(f"L1 lambda: {args.l1_lambda}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    asyncio.run(main())