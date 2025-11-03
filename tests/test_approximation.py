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
    parser.add_argument("--user_id", type=int, default=0)
    # parser.add_argument("--test-data", type=str, default="data/high_variance_questions/user1_test.json", help="Test data path")
    parser.add_argument("--l1-lambda", type=float, default=1, help="L1 regularization parameter")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8080", help="Gateway URL")
    
    args = parser.parse_args()

    train_data_path = f"data/PERSONA_testing/user{args.user_id}_train.json"
    
    with open("configs/attribute_prompts_400.json", "r") as f:
        attribute_prompts = json.load(f)["prompts"]

    selected_attributes =  [6, 24, 1, 21, 22, 22, 12, 13]
    selected_attributes = [attribute_prompts[i] for i in selected_attributes]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_prompt = "You are an AI assistant."
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    print(f"Loading training data from {train_data_path}")
    with open(train_data_path, 'r') as f:
        train_data_raw = json.load(f)

    other_users = random.sample(list(range(0, 150)), 10)

    total_data = [] 
    for user_id in other_users:
        other_user_path = f"data/PERSONA_testing/user{user_id}_train.json"
        with open(other_user_path, 'r') as f:
            other_user_data = json.load(f)
        for i in range(len(train_data_raw)):
            curr = train_data_raw[i]
            curr['rejected'] = other_user_data[i]['rejected'] 
            total_data.append(curr)

    # Convert to drift format for approximate function: list of (prompt, chosen, rejected) tuples
    train_data_tuples = []
    for i, item in enumerate(total_data):
        train_data_tuples.append((item['prompt'], item['chosen'], item['rejected']))

    num_train = int(0.8 * len(total_data)) 
    test_data_tuples = train_data_tuples[num_train:]
    train_data_tuples = train_data_tuples[:num_train]

    
    # Convert test data to dict format for evaluate_accuracy function
    test_data_dicts = []
    for prompt, chosen, rejected in test_data_tuples:
        test_data_dicts.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
    
    print(f"Loaded {len(train_data_tuples)} training samples")
    
    # Find p vector
    print("Finding p vector...")
    p = await approximate(args.gateway_url, train_data_tuples, tokenizer, MODEL_ID, base_prompt, selected_attributes, args.l1_lambda)
    
    print(f"Found p vector with {np.count_nonzero(p)} non-zero components")
    print(f"P vector norm: {np.linalg.norm(p):.4f}")
    print(f"Top 5 attributes by weight:")
    top_indices = np.argsort(np.abs(p))[-5:][::-1]
    for i in top_indices:
        print(f"  {i}: {p[i]:.4f} - {selected_attributes[i][:80]}...")
    
    # Evaluate accuracy on test data
    print(f"Evaluating accuracy on test data...")
    accuracy = await evaluate_accuracy(args.gateway_url, test_data_dicts, p, tokenizer, MODEL_ID, base_prompt, selected_attributes)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    asyncio.run(main())