import json

with open('../configs/attribute_prompts_400.json', 'r') as f:
    attribute_prompts = json.load(f)['prompts']

selected_indicies = [155, 160, 1, 225, 361, 368, 393, 315, 109, 386, 22, 352, 243, 141, 294, 332, 319, 79, 88, 363]
selected_attributes = [attribute_prompts[i] for i in selected_indicies]

random_attributes = random.sample(attribute_prompts, len(selected_attributes))

import asyncio
import json
import numpy as np
import argparse
from transformers import AutoTokenizer

# Import async utils for log prob computation and drift approximation
from gumbel.utils.async_utils import approximate_async, evaluate_accuracy_async, MODEL_ID

# Import attribute prompts - support both local import and config file
import sys
import os
sys.path.append('../utils')

async def main():
    parser = argparse.ArgumentParser(description="Test preference approximation using async VLLM")
    parser.add_argument("--train-data", type=str, default="data/persona_pref/user11_train.json", help="Training data path")
    parser.add_argument("--test-data", type=str, default="data/persona_pref/user11_test.json", help="Test data path")
    parser.add_argument("--max-train-samples", type=int, default=150, help="Max training samples")
    parser.add_argument("--max-attributes", type=int, default=50, help="Max attribute prompts to use")
    parser.add_argument("--l1-lambda", type=float, default=0.01, help="L1 regularization parameter")
    parser.add_argument("--attribute-config", type=str, help="Path to attribute prompts JSON config file")
    parser.add_argument("--selected-config", type=str, help="Path to selected.py file with indices")
    parser.add_argument("--sparsity-key", type=str, choices=["2e-5", "3e-5", "4e-5"], default="3e-5", help="Sparsity level key for selected indices")
    
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
    
    # Load attribute prompts from config file if provided
    if args.attribute_config:
        print(f"Loading attribute prompts from config: {args.attribute_config}")
        config_prompts = load_attribute_prompts_from_config(args.attribute_config)
        print(f"Loaded {len(config_prompts)} prompts from config file")
        available_prompts = config_prompts
    else:
        print("Using default attribute prompts from import")
        available_prompts = attribute_prompts
    
    # Use selected indices if provided
    if args.selected_config:
        print(f"Loading selected indices from: {args.selected_config} (key: {args.sparsity_key})")
        try:
            selected_indices = load_selected_indices(args.selected_config, args.sparsity_key)
            print(f"Loaded {len(selected_indices)} selected indices: {selected_indices}")
            
            # Select attributes using the indices
            selected_attributes = [available_prompts[i] for i in selected_indices if i < len(available_prompts)]
            print(f"Selected {len(selected_attributes)} attributes using indices")
            
            # Warn if some indices were out of range
            invalid_indices = [i for i in selected_indices if i >= len(available_prompts)]
            if invalid_indices:
                print(f"⚠️  Warning: {len(invalid_indices)} indices out of range (max index: {len(available_prompts)-1}): {invalid_indices}")
                
        except Exception as e:
            print(f"❌ Failed to load selected indices: {e}")
            print("Falling back to max_attributes selection")
            selected_attributes = available_prompts[:args.max_attributes]
    else:
        # Use subset of attribute prompts (original behavior)
        selected_attributes = available_prompts[:args.max_attributes]
        print(f"Using first {len(selected_attributes)} attribute prompts")
    
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
    
    if args.selected_config:
        print(f"\nConfiguration:")
        print(f"Selected indices source: {args.selected_config}")
        print(f"Sparsity key: {args.sparsity_key}")
        print(f"Total available attributes: {len(available_prompts)}")
        print(f"Selected attributes: {len(selected_attributes)}")
    
    if args.attribute_config:
        print(f"Attribute prompts source: {args.attribute_config}")


if __name__ == "__main__":
    asyncio.run(main())