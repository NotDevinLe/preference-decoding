#!/usr/bin/env python3
"""
Test Approximation Sweep Script
Runs approximation tests across all sparsity keys from selected.py with varying L1 lambda values.
Uses the attribute_prompts.json config file for all tests.

This script supports both single-user and multi-user modes.

This script will test all combinations of:
- Sparsity keys: ["2e-5", "3e-5", "4e-5"] (from selected.py)
- L1 lambda values: [0.001, 0.01, 0.1, 1.0] (or custom range)

Usage examples:

    # Single user mode: Run sweep with default lambda values
    python test_approximation_sweep.py --train-data data/user1_train.json --test-data data/user1_test.json

    # Single user mode: Custom lambda range
    python test_approximation_sweep.py --train-data data/user1_train.json --test-data data/user1_test.json --lambda-values 0.001 0.01 0.05 0.1

    # Single user mode: Single lambda value across all sparsity levels
    python test_approximation_sweep.py --train-data data/user1_train.json --test-data data/user1_test.json --lambda-values 0.01

    # Multi-user mode: Process user range 
    python test_approximation_sweep.py --users user1-5 --data-dir data/persona_pref/

    # Multi-user mode: Process specific users
    python test_approximation_sweep.py --users user1,user3,user5 --data-dir data/persona_pref/

    # Multi-user mode: Custom parameters
    python test_approximation_sweep.py --users user1-3 --data-dir data/persona_pref/ --lambda-values 0.01 0.1 --sparsity-keys 3e-5 4e-5

    # Save results to CSV (works for both modes)
    python test_approximation_sweep.py --users user1-5 --output multi_user_results.csv
"""

import asyncio
import json
import numpy as np
import argparse
import csv
import os
import sys
from transformers import AutoTokenizer
from datetime import datetime

# Import async utils for log prob computation and drift approximation
from async_utils import approximate_async, evaluate_accuracy_async, MODEL_ID

# Add configs directory to path
sys.path.insert(0, '../configs')

def load_attribute_prompts_from_config(config_path):
    """Load attribute prompts from JSON config file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data["prompts"]

def load_selected_indices(selected_path, sparsity_key):
    """Load selected indices from selected.py"""
    sys.path.insert(0, os.path.dirname(selected_path))
    import selected
    
    if sparsity_key in selected.selected_ind:
        return selected.selected_ind[sparsity_key]
    else:
        available_keys = list(selected.selected_ind.keys())
        raise ValueError(f"Sparsity key '{sparsity_key}' not found. Available keys: {available_keys}")

async def run_single_test(train_data, test_data, tokenizer, base_prompt, selected_attributes, l1_lambda, sparsity_key):
    """Run a single approximation test and return results"""
    print(f"\n🔧 TESTING: Sparsity={sparsity_key}, L1={l1_lambda}, Attributes={len(selected_attributes)}")
    
    # Find p vector
    try:
        p = await approximate_async(train_data, tokenizer, base_prompt, selected_attributes, l1_lambda)
        
        # Evaluate accuracy
        accuracy = await evaluate_accuracy_async(test_data, p, tokenizer, base_prompt, selected_attributes)
        
        # Calculate statistics
        non_zero_components = np.count_nonzero(p)
        p_norm = np.linalg.norm(p)
        sparsity_ratio = non_zero_components / len(p) if len(p) > 0 else 0
        
        # Get top attributes
        top_indices = np.argsort(np.abs(p))[-5:][::-1]
        top_weights = [p[i] for i in top_indices]
        top_attributes = [selected_attributes[i][:50] + "..." for i in top_indices]
        
        result = {
            'sparsity_key': sparsity_key,
            'l1_lambda': l1_lambda,
            'num_attributes': len(selected_attributes),
            'accuracy': accuracy,
            'non_zero_components': non_zero_components,
            'sparsity_ratio': sparsity_ratio,
            'p_norm': p_norm,
            'top_weights': top_weights,
            'top_attributes': top_attributes,
            'success': True,
            'error': None
        }
        
        print(f"✅ SUCCESS: Accuracy={accuracy:.4f} ({accuracy*100:.2f}%), Non-zero={non_zero_components}/{len(selected_attributes)}")
        return result
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {
            'sparsity_key': sparsity_key,
            'l1_lambda': l1_lambda,
            'num_attributes': len(selected_attributes),
            'accuracy': 0.0,
            'non_zero_components': 0,
            'sparsity_ratio': 0.0,
            'p_norm': 0.0,
            'top_weights': [],
            'top_attributes': [],
            'success': False,
            'error': str(e)
        }

def save_results_to_csv(results, output_path):
    """Save results to CSV file"""
    if not results:
        print("⚠️  No results to save")
        return
    
    # Check if we have user information (multi-user mode)
    has_user_info = any('user' in result for result in results)
    
    fieldnames = [
        'sparsity_key', 'l1_lambda', 'num_attributes', 'accuracy', 
        'non_zero_components', 'sparsity_ratio', 'p_norm', 'success', 'error',
        'top_weight_1', 'top_weight_2', 'top_weight_3', 'top_weight_4', 'top_weight_5',
        'top_attr_1', 'top_attr_2', 'top_attr_3', 'top_attr_4', 'top_attr_5'
    ]
    
    # Add user field if we're in multi-user mode
    if has_user_info:
        fieldnames.insert(0, 'user')
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Flatten top weights and attributes
            row = {k: v for k, v in result.items() if k not in ['top_weights', 'top_attributes']}
            
            # Add top weights (pad with None if fewer than 5)
            top_weights = result['top_weights'] + [None] * (5 - len(result['top_weights']))
            for i, weight in enumerate(top_weights[:5]):
                row[f'top_weight_{i+1}'] = weight
            
            # Add top attributes (pad with None if fewer than 5)
            top_attrs = result['top_attributes'] + [None] * (5 - len(result['top_attributes']))
            for i, attr in enumerate(top_attrs[:5]):
                row[f'top_attr_{i+1}'] = attr
            
            writer.writerow(row)
    
    print(f"💾 Results saved to: {output_path}")

def print_summary_table(results):
    """Print a summary table of all results"""
    if not results:
        print("⚠️  No results to summarize")
        return
    
    # Check if we have user information (multi-user mode)
    has_user_info = any('user' in result for result in results)
    
    print("\n" + "="*90)
    print("SUMMARY RESULTS")
    print("="*90)
    
    if has_user_info:
        print(f"{'User':<8} {'Sparsity':<10} {'L1 Lambda':<10} {'Attributes':<12} {'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 90)
        
        for result in results:
            user_name = result.get('user', 'N/A')
            if result['success']:
                print(f"{user_name:<8} {result['sparsity_key']:<10} {result['l1_lambda']:<10.3f} {result['num_attributes']:<12} "
                      f"{result['accuracy']:<10.4f} {result['non_zero_components']:<10} {result['p_norm']:<10.4f}")
            else:
                print(f"{user_name:<8} {result['sparsity_key']:<10} {result['l1_lambda']:<10.3f} {result['num_attributes']:<12} "
                      f"{'FAILED':<10} {'N/A':<10} {'N/A':<10}")
    else:
        print(f"{'Sparsity':<10} {'L1 Lambda':<10} {'Attributes':<12} {'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 80)
        
        for result in results:
            if result['success']:
                print(f"{result['sparsity_key']:<10} {result['l1_lambda']:<10.3f} {result['num_attributes']:<12} "
                      f"{result['accuracy']:<10.4f} {result['non_zero_components']:<10} {result['p_norm']:<10.4f}")
            else:
                print(f"{result['sparsity_key']:<10} {result['l1_lambda']:<10.3f} {result['num_attributes']:<12} "
                      f"{'FAILED':<10} {'N/A':<10} {'N/A':<10}")
    
    # Find best results
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_accuracy = max(successful_results, key=lambda x: x['accuracy'])
        if has_user_info:
            print(f"\n🏆 BEST ACCURACY: {best_accuracy['accuracy']:.4f} "
                  f"(User: {best_accuracy.get('user', 'N/A')}, Sparsity: {best_accuracy['sparsity_key']}, L1: {best_accuracy['l1_lambda']})")
        else:
            print(f"\n🏆 BEST ACCURACY: {best_accuracy['accuracy']:.4f} "
                  f"(Sparsity: {best_accuracy['sparsity_key']}, L1: {best_accuracy['l1_lambda']})")
        
        # If multi-user mode, also show per-user best results
        if has_user_info:
            print(f"\n📊 PER-USER BEST RESULTS:")
            users = set(r.get('user', 'N/A') for r in successful_results)
            for user in sorted(users):
                user_results = [r for r in successful_results if r.get('user') == user]
                if user_results:
                    best_user = max(user_results, key=lambda x: x['accuracy'])
                    print(f"  {user}: {best_user['accuracy']:.4f} "
                          f"(Sparsity: {best_user['sparsity_key']}, L1: {best_user['l1_lambda']})")

def parse_user_range(user_range_str):
    """Parse user range string like 'user1-5' or 'user1,user3,user5' into list of user names"""
    if not user_range_str:
        return []
    
    users = []
    for part in user_range_str.split(','):
        part = part.strip()
        if '-' in part and part.startswith('user'):
            # Handle range like 'user1-5'
            prefix = 'user'
            range_part = part[4:]  # Remove 'user' prefix
            if '-' in range_part:
                start, end = range_part.split('-')
                try:
                    start_num = int(start)
                    end_num = int(end)
                    for i in range(start_num, end_num + 1):
                        users.append(f"{prefix}{i}")
                except ValueError:
                    users.append(part)  # If parsing fails, treat as literal
            else:
                users.append(part)
        else:
            users.append(part)
    
    return users

async def run_user_sweep(user_name, data_dir, tokenizer, base_prompt, available_prompts, args):
    """Run sweep for a single user"""
    print(f"\n👤 PROCESSING USER: {user_name}")
    
    # Construct file paths
    train_path = os.path.join(data_dir, f"{user_name}_train.json")
    test_path = os.path.join(data_dir, f"{user_name}_test.json")
    
    # Check if files exist
    if not os.path.exists(train_path):
        print(f"❌ Training file not found: {train_path}")
        return []
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return []
    
    # Load training data
    try:
        with open(train_path, 'r') as f:
            train_data_raw = json.load(f)
        
        # Convert to drift format
        train_data = []
        for item in train_data_raw[:args.max_train_samples]:
            train_data.append((item['prompt'], item['chosen'], item['rejected']))
        
        print(f"Loaded {len(train_data)} training samples")
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return []
    
    # Load test data
    try:
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return []
    
    # Run tests for all combinations
    user_results = []
    total_tests = len(args.sparsity_keys) * len(args.lambda_values)
    current_test = 0
    
    for sparsity_key in args.sparsity_keys:
        print(f"\n  📊 SPARSITY LEVEL: {sparsity_key}")
        
        # Load selected indices for this sparsity level
        try:
            selected_indices = load_selected_indices(args.selected_config, sparsity_key)
            selected_attributes = [available_prompts[i] for i in selected_indices if i < len(available_prompts)]
            
            invalid_indices = [i for i in selected_indices if i >= len(available_prompts)]
            if invalid_indices:
                print(f"  ⚠️  Warning: {len(invalid_indices)} indices out of range")
            
            print(f"  Selected {len(selected_attributes)} attributes using {sparsity_key} indices")
            
        except Exception as e:
            print(f"  ❌ Failed to load selected indices for {sparsity_key}: {e}")
            continue
        
        # Test each lambda value
        for l1_lambda in args.lambda_values:
            current_test += 1
            print(f"\n  [{current_test}/{total_tests}] {user_name}", end=" ")
            
            result = await run_single_test(
                train_data, test_data, tokenizer, base_prompt, 
                selected_attributes, l1_lambda, sparsity_key
            )
            # Add user information to result
            result['user'] = user_name
            user_results.append(result)
    
    return user_results

async def main():
    parser = argparse.ArgumentParser(description="Test preference approximation across all sparsity levels and L1 values")
    
    # Single user mode (original behavior)
    parser.add_argument("--train-data", type=str, help="Training data path (single user mode)")
    parser.add_argument("--test-data", type=str, help="Test data path (single user mode)")
    
    # Multi-user mode
    parser.add_argument("--data-dir", type=str, default="data/persona_pref/", help="Directory containing user data files")
    parser.add_argument("--users", type=str, help="User range (e.g., 'user1-5' or 'user1,user3,user5')")
    
    # Common parameters
    parser.add_argument("--max-train-samples", type=int, default=150, help="Max training samples")
    parser.add_argument("--lambda-values", type=float, nargs='+', default=[0.001, 0.01, 0.1, 1.0], help="L1 lambda values to test")
    parser.add_argument("--sparsity-keys", type=str, nargs='+', choices=["2e-5", "3e-5", "4e-5"], default=["2e-5", "3e-5", "4e-5"], help="Sparsity keys to test")
    parser.add_argument("--output", type=str, help="Output CSV file path (optional)")
    parser.add_argument("--attribute-config", type=str, default="../configs/attribute_prompts.json", help="Path to attribute prompts JSON")
    parser.add_argument("--selected-config", type=str, default="../configs/selected.py", help="Path to selected.py file")
    
    args = parser.parse_args()
    
    # Determine mode: single user or multi-user
    if args.users:
        # Multi-user mode
        user_list = parse_user_range(args.users)
        print(f"🚀 MULTI-USER APPROXIMATION SWEEP")
        print(f"Data directory: {args.data_dir}")
        print(f"Users: {user_list}")
        print(f"Sparsity keys: {args.sparsity_keys}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Max training samples per user: {args.max_train_samples}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        base_prompt = "You are a helpful assistant."
        
        # Load attribute prompts
        print(f"\n📝 Loading attribute prompts from {args.attribute_config}")
        available_prompts = load_attribute_prompts_from_config(args.attribute_config)
        print(f"Loaded {len(available_prompts)} total attribute prompts")
        
        # Process each user
        all_results = []
        for user_name in user_list:
            user_results = await run_user_sweep(user_name, args.data_dir, tokenizer, base_prompt, available_prompts, args)
            all_results.extend(user_results)
        
        results = all_results
        
    else:
        # Single user mode (original behavior)
        if not args.train_data or not args.test_data:
            print("❌ Error: For single user mode, both --train-data and --test-data are required")
            return
        
        print(f"🚀 SINGLE-USER APPROXIMATION SWEEP")
        print(f"Training data: {args.train_data}")
        print(f"Test data: {args.test_data}")
        print(f"Sparsity keys: {args.sparsity_keys}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Max training samples: {args.max_train_samples}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        base_prompt = "You are a helpful assistant."
        
        # Load training data
        print(f"\n📚 Loading training data from {args.train_data}")
        with open(args.train_data, 'r') as f:
            train_data_raw = json.load(f)
        
        # Convert to drift format
        train_data = []
        for item in train_data_raw[:args.max_train_samples]:
            train_data.append((item['prompt'], item['chosen'], item['rejected']))
        
        print(f"Loaded {len(train_data)} training samples")
        
        # Load test data
        print(f"📚 Loading test data from {args.test_data}")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Load attribute prompts
        print(f"\n📝 Loading attribute prompts from {args.attribute_config}")
        available_prompts = load_attribute_prompts_from_config(args.attribute_config)
        print(f"Loaded {len(available_prompts)} total attribute prompts")
        
        # Run tests for all combinations
        results = []
        total_tests = len(args.sparsity_keys) * len(args.lambda_values)
        current_test = 0
        
        print(f"\n🧪 Running {total_tests} total tests...")
        
        for sparsity_key in args.sparsity_keys:
            print(f"\n📊 SPARSITY LEVEL: {sparsity_key}")
            
            # Load selected indices for this sparsity level
            try:
                selected_indices = load_selected_indices(args.selected_config, sparsity_key)
                selected_attributes = [available_prompts[i] for i in selected_indices if i < len(available_prompts)]
                
                invalid_indices = [i for i in selected_indices if i >= len(available_prompts)]
                if invalid_indices:
                    print(f"⚠️  Warning: {len(invalid_indices)} indices out of range")
                
                print(f"Selected {len(selected_attributes)} attributes using {sparsity_key} indices")
                
            except Exception as e:
                print(f"❌ Failed to load selected indices for {sparsity_key}: {e}")
                continue
            
            # Test each lambda value
            for l1_lambda in args.lambda_values:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}]", end=" ")
                
                result = await run_single_test(
                    train_data, test_data, tokenizer, base_prompt, 
                    selected_attributes, l1_lambda, sparsity_key
                )
                results.append(result)
    
    # Print summary
    print_summary_table(results)
    
    # Save to CSV if requested
    if args.output:
        save_results_to_csv(results, args.output)
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.users:
            user_suffix = args.users.replace(",", "_").replace("-", "_")
            auto_output = f"approximation_sweep_multi_{user_suffix}_{timestamp}.csv"
        else:
            auto_output = f"approximation_sweep_single_{timestamp}.csv"
        save_results_to_csv(results, auto_output)
    
    print(f"\n✅ SWEEP COMPLETE: {len([r for r in results if r['success']])}/{len(results)} tests successful")

if __name__ == "__main__":
    asyncio.run(main())