#!/usr/bin/env python3
"""
Test Approximation Random/Provided Selection Script
- RANDOM mode (default): randomly choose attributes of requested sizes.
- PROVIDED mode: pass --features-json pointing to a JSON file with entries like
  {"sparsity": 0.0001, "lr": 0.01, "features": [2,3,4,...]} (only 'features' is used).

Works for single-user and multi-user; saves CSV; prints summary.
"""

import asyncio
import json
import numpy as np
import argparse
import csv
import os
import sys
import random
from transformers import AutoTokenizer
from datetime import datetime

from gumbel.utils.async_utils import approximate_async, evaluate_accuracy_async, MODEL_ID

# Add configs directory to path
sys.path.insert(0, '../configs')

def load_attribute_prompts_from_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data["prompts"]

def select_random_attributes(available_prompts, num_attributes, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if num_attributes >= len(available_prompts):
        return available_prompts.copy()
    indices = random.sample(range(len(available_prompts)), num_attributes)
    indices.sort()
    return [available_prompts[i] for i in indices]

def load_feature_sets_json(features_json_path):
    """
    Load provided feature sets and bucket them by size.
    Returns: dict[int, list[dict(id=<int>, features=<list[int]>)]] and flat list
    """
    with open(features_json_path, 'r') as f:
        data = json.load(f)
    by_size = {}
    flat = []
    for idx, entry in enumerate(data):
        feats = entry.get("features", [])
        if not isinstance(feats, list):
            continue
        n = len(feats)
        item = {"id": idx, "features": feats}
        by_size.setdefault(n, []).append(item)
        flat.append(item)
    return by_size, flat

async def run_single_test(train_data, test_data, tokenizer, base_prompt,
                          selected_attributes, l1_lambda, num_attributes, test_id, selection_type):
    print(f"\n🔧 TESTING: {selection_type}={num_attributes} attrs, L1={l1_lambda}, Test ID={test_id}")
    try:
        p = await approximate_async(train_data, tokenizer, base_prompt, selected_attributes, l1_lambda)
        accuracy = await evaluate_accuracy_async(test_data, p, tokenizer, base_prompt, selected_attributes)

        non_zero_components = int(np.count_nonzero(p))
        p_norm = float(np.linalg.norm(p))
        sparsity_ratio = (non_zero_components / float(len(p))) if len(p) > 0 else 0.0

        top_idx = np.argsort(np.abs(p))[-5:][::-1]
        top_weights = [float(p[i]) for i in top_idx]
        top_attributes = []
        for i in top_idx:
            s = selected_attributes[i]
            if len(s) > 50:
                s = s[:50] + "..."
            top_attributes.append(s)

        result = {
            'selection_type': selection_type,
            'num_attributes': int(num_attributes),
            'test_id': str(test_id),
            'l1_lambda': float(l1_lambda),
            'actual_attributes': len(selected_attributes),
            'accuracy': float(accuracy),
            'non_zero_components': non_zero_components,
            'sparsity_ratio': float(sparsity_ratio),
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
            'selection_type': selection_type,
            'num_attributes': int(num_attributes),
            'test_id': str(test_id),
            'l1_lambda': float(l1_lambda),
            'actual_attributes': len(selected_attributes),
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
    if not results:
        print("⚠️  No results to save")
        return

    has_user_info = any('user' in r for r in results)
    fieldnames = [
        'selection_type', 'num_attributes', 'test_id', 'l1_lambda', 'actual_attributes', 'accuracy',
        'non_zero_components', 'sparsity_ratio', 'p_norm', 'success', 'error',
        'top_weight_1', 'top_weight_2', 'top_weight_3', 'top_weight_4', 'top_weight_5',
        'top_attr_1', 'top_attr_2', 'top_attr_3', 'top_attr_4', 'top_attr_5'
    ]
    if has_user_info:
        fieldnames.insert(0, 'user')

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k not in ['top_weights', 'top_attributes']}
            tw = r.get('top_weights', [])[:5] + [None] * (5 - len(r.get('top_weights', [])))
            ta = r.get('top_attributes', [])[:5] + [None] * (5 - len(r.get('top_attributes', [])))
            for i in range(5):
                row[f'top_weight_{i+1}'] = tw[i]
                row[f'top_attr_{i+1}'] = ta[i]
            writer.writerow(row)

    print(f"💾 Results saved to: {output_path}")

def print_summary_table(results):
    if not results:
        print("⚠️  No results to summarize")
        return

    has_user_info = any('user' in r for r in results)
    print("\n" + "="*100)
    print("SUMMARY RESULTS - RANDOM / PROVIDED SELECTION")
    print("="*100)

    if has_user_info:
        print(f"{'User':<8} {'SelType':<9} {'Attrs':<7} {'Test ID':<8} {'L1 Lambda':<10} {'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 100)
        for r in results:
            if r['success']:
                print(f"{r.get('user','N/A'):<8} {r['selection_type']:<9} {r['num_attributes']:<7} {r['test_id']:<8} "
                      f"{r['l1_lambda']:<10.3f} {r['accuracy']:<10.4f} {r['non_zero_components']:<10} {r['p_norm']:<10.4f}")
            else:
                print(f"{r.get('user','N/A'):<8} {r['selection_type']:<9} {r['num_attributes']:<7} {r['test_id']:<8} "
                      f"{r['l1_lambda']:<10.3f} {'FAILED':<10} {'N/A':<10} {'N/A':<10}")
    else:
        print(f"{'SelType':<9} {'Attrs':<7} {'Test ID':<8} {'L1 Lambda':<10} {'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 90)
        for r in results:
            if r['success']:
                print(f"{r['selection_type']:<9} {r['num_attributes']:<7} {r['test_id']:<8} {r['l1_lambda']:<10.3f} "
                      f"{r['accuracy']:<10.4f} {r['non_zero_components']:<10} {r['p_norm']:<10.4f}")
            else:
                print(f"{r['selection_type']:<9} {r['num_attributes']:<7} {r['test_id']:<8} {r['l1_lambda']:<10.3f} "
                      f"{'FAILED':<10} {'N/A':<10} {'N/A':<10}")

    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda x: x['accuracy'])
        if has_user_info:
            print(f"\n🏆 BEST ACCURACY: {best['accuracy']:.4f} "
                  f"(User: {best.get('user','N/A')}, Type: {best['selection_type']}, Attrs: {best['num_attributes']}, "
                  f"L1: {best['l1_lambda']}, Test: {best['test_id']})")
        else:
            print(f"\n🏆 BEST ACCURACY: {best['accuracy']:.4f} "
                  f"(Type: {best['selection_type']}, Attrs: {best['num_attributes']}, "
                  f"L1: {best['l1_lambda']}, Test: {best['test_id']})")

def parse_user_range(user_range_str):
    if not user_range_str:
        return []
    users = []
    for part in user_range_str.split(','):
        part = part.strip()
        if '-' in part and part.startswith('user'):
            prefix = 'user'
            range_part = part[len(prefix):]
            if '-' in range_part:
                start, end = range_part.split('-')
                try:
                    start_num = int(start); end_num = int(end)
                    for i in range(start_num, end_num + 1):
                        users.append(f"{prefix}{i}")
                except ValueError:
                    users.append(part)
            else:
                users.append(part)
        else:
            users.append(part)
    return users

async def run_user_sweep(user_name, data_dir, tokenizer, base_prompt, available_prompts, args, provided_by_size):
    print(f"\n👤 PROCESSING USER: {user_name}")
    train_path = os.path.join(data_dir, f"{user_name}_train.json")
    test_path  = os.path.join(data_dir, f"{user_name}_test.json")

    if not os.path.exists(train_path):
        print(f"❌ Training file not found: {train_path}")
        return []
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return []

    # Load training data
    with open(train_path, 'r') as f:
        train_raw = json.load(f)
    train_data = [(it['prompt'], it['chosen'], it['rejected']) for it in train_raw[:args.max_train_samples]]
    print(f"Loaded {len(train_data)} training samples")

    # Load test data
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    results = []

    if args.features_json:
        # PROVIDED mode
        total_tests = sum(len(provided_by_size.get(n, [])[:1 if args.provided_first_only else None]) * len(args.lambda_values)
                          for n in args.attribute_counts)
        current = 0
        print(f"\n🧪 Running {total_tests} total tests (PROVIDED)...")
        for n in args.attribute_counts:
            sets = provided_by_size.get(n, [])
            if not sets:
                print(f"⚠️  No provided feature set of size {n}; skipping")
                continue
            use_sets = sets[:1] if args.provided_first_only else sets
            print(f"\n📊 ATTRIBUTE COUNT (provided): {n} (sets: {len(use_sets)}/{len(sets)})")
            for s in use_sets:
                # Map indices to prompts (bounds check)
                selected_attributes = [available_prompts[i] for i in s['features'] if 0 <= i < len(available_prompts)]
                for l1 in args.lambda_values:
                    current += 1
                    test_id = f"provided_{s['id']}"
                    print(f"\n[{current}/{total_tests}] {user_name} {test_id}", end=" ")
                    res = await run_single_test(train_data, test_data, tokenizer, base_prompt,
                                                selected_attributes, l1, n, test_id, 'provided')
                    res['user'] = user_name
                    results.append(res)
    else:
        # RANDOM mode
        total_tests = len(args.attribute_counts) * len(args.lambda_values) * args.random_runs
        current = 0
        print(f"\n🧪 Running {total_tests} total tests (RANDOM)...")
        for n in args.attribute_counts:
            print(f"\n📊 ATTRIBUTE COUNT (random): {n}")
            for run_id in range(args.random_runs):
                # deterministic per (user, n, run) if seed provided
                run_seed = None
                if args.random_seed is not None:
                    run_seed = args.random_seed + (hash(f"{user_name}_{n}_{run_id}") % 10000)
                selected_attributes = select_random_attributes(available_prompts, n, run_seed)
                print(f"🎲 Selected {len(selected_attributes)} random attributes (run {run_id+1}/{args.random_runs})")
                for l1 in args.lambda_values:
                    current += 1
                    test_id = f"{run_id + 1}"
                    print(f"\n[{current}/{total_tests}] {user_name} run {test_id}", end=" ")
                    res = await run_single_test(train_data, test_data, tokenizer, base_prompt,
                                                selected_attributes, l1, n, test_id, 'random')
                    res['user'] = user_name
                    results.append(res)

    return results

async def main():
    parser = argparse.ArgumentParser(description="Test preference approximation using RANDOM or PROVIDED attribute selection")

    # Single user
    parser.add_argument("--train-data", type=str, help="Training data path (single user mode)")
    parser.add_argument("--test-data",  type=str, help="Test data path (single user mode)")

    # Multi-user
    parser.add_argument("--data-dir", type=str, default="data/persona_pref/", help="Directory containing user data files")
    parser.add_argument("--users", type=str, help="User range (e.g., 'user1-5' or 'user1,user3,user5')")

    # Common
    parser.add_argument("--max-train-samples", type=int, default=150, help="Max training samples")
    parser.add_argument("--lambda-values", type=float, nargs='+', default=[0.01], help="L1 lambda values to test")
    parser.add_argument("--attribute-counts", type=int, nargs='+', default=[5, 23, 194], help="Numbers of attributes to test")
    parser.add_argument("--random-runs", type=int, default=3, help="Number of random runs per attribute count")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducible results")

    # PROVIDED selection mode
    parser.add_argument("--features-json", type=str, help="Path to JSON with entries containing 'features': [...]. If set, uses PROVIDED mode")
    parser.add_argument("--provided-first-only", action="store_true", help="Use only the first matching feature set per attribute count")

    # Output/config
    parser.add_argument("--output", type=str, help="Output CSV file path (optional)")
    parser.add_argument("--attribute-config", type=str, default="gumbel/configs/attribute_prompts.json", help="Path to attribute prompts JSON")

    args = parser.parse_args()

    # Seeds (for RANDOM mode and any per-run seeding we do)
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        print(f"🎲 Using base random seed: {args.random_seed}")

    # Tokenizer & prompts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_prompt = "You are a helpful assistant."
    print(f"\n📝 Loading attribute prompts from {args.attribute_config}")
    available_prompts = load_attribute_prompts_from_config(args.attribute_config)
    print(f"Loaded {len(available_prompts)} total attribute prompts")

    # Provided selection map (if enabled)
    provided_by_size = {}
    if args.features_json:
        print(f"📦 Loading provided feature sets from {args.features_json}")
        provided_by_size, _ = load_feature_sets_json(args.features_json)
        print(f"   Found {len(provided_by_size)} unique sizes")

    # Mode: multi-user or single-user
    if args.users:
        user_list = parse_user_range(args.users)
        print(f"\n🚀 MULTI-USER SWEEP")
        print(f"Users: {user_list}")
        print(f"Attribute counts: {args.attribute_counts}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Random runs per count: {args.random_runs}")
        print(f"Max training samples per user: {args.max_train_samples}")
        if args.features_json:
            print("Mode: PROVIDED (from features JSON)")
            if args.provided_first_only:
                print("Note: Using only the first matching set per size.")
        else:
            print("Mode: RANDOM")

        all_results = []
        for user_name in user_list:
            user_results = await run_user_sweep(user_name, args.data_dir, tokenizer, base_prompt,
                                                available_prompts, args, provided_by_size)
            all_results.extend(user_results)
        results = all_results

    else:
        # Single-user mode requires both paths
        if not args.train_data or not args.test_data:
            print("❌ Error: For single user mode, both --train-data and --test-data are required")
            return

        print(f"\n🚀 SINGLE-USER SWEEP")
        print(f"Training data: {args.train_data}")
        print(f"Test data: {args.test_data}")
        print(f"Attribute counts: {args.attribute_counts}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Random runs per count: {args.random_runs}")
        print(f"Max training samples: {args.max_train_samples}")
        print("Mode:", "PROVIDED" if args.features_json else "RANDOM")
        if args.features_json and args.provided_first_only:
            print("Note: Using only the first matching set per size.")

        with open(args.train_data, 'r') as f:
            train_raw = json.load(f)
        train_data = [(it['prompt'], it['chosen'], it['rejected']) for it in train_raw[:args.max_train_samples]]
        print(f"Loaded {len(train_data)} training samples")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")

        results = []
        if args.features_json:
            # PROVIDED mode
            total_tests = sum(len(provided_by_size.get(n, [])[:1 if args.provided_first_only else None]) * len(args.lambda_values)
                              for n in args.attribute_counts)
            current = 0
            print(f"\n🧪 Running {total_tests} total tests (PROVIDED)...")
            for n in args.attribute_counts:
                sets = provided_by_size.get(n, [])
                if not sets:
                    print(f"⚠️  No provided feature set of size {n}; skipping")
                    continue
                use_sets = sets[:1] if args.provided_first_only else sets
                print(f"\n📊 ATTRIBUTE COUNT (provided): {n} (sets: {len(use_sets)}/{len(sets)})")
                for s in use_sets:
                    selected_attributes = [available_prompts[i] for i in s['features'] if 0 <= i < len(available_prompts)]
                    for l1 in args.lambda_values:
                        current += 1
                        test_id = f"provided_{s['id']}"
                        print(f"\n[{current}/{total_tests}] {test_id}", end=" ")
                        res = await run_single_test(train_data, test_data, tokenizer, base_prompt,
                                                    selected_attributes, l1, n, test_id, 'provided')
                        results.append(res)
        else:
            # RANDOM mode
            total_tests = len(args.attribute_counts) * len(args.lambda_values) * args.random_runs
            current = 0
            print(f"\n🧪 Running {total_tests} total tests (RANDOM)...")
            for n in args.attribute_counts:
                print(f"\n📊 ATTRIBUTE COUNT (random): {n}")
                for run_id in range(args.random_runs):
                    run_seed = None
                    if args.random_seed is not None:
                        run_seed = args.random_seed + (hash(f"{n}_{run_id}") % 10000)
                    selected_attributes = select_random_attributes(available_prompts, n, run_seed)
                    print(f"🎲 Selected {len(selected_attributes)} random attributes (run {run_id+1}/{args.random_runs})")
                    for l1 in args.lambda_values:
                        current += 1
                        test_id = f"{run_id + 1}"
                        print(f"\n[{current}/{total_tests}] Run {test_id}", end=" ")
                        res = await run_single_test(train_data, test_data, tokenizer, base_prompt,
                                                    selected_attributes, l1, n, test_id, 'random')
                        results.append(res)

    # Summary + save
    print_summary_table(results)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "multi" if args.users else "single"
        mode = "provided" if args.features_json else "random"
        output_path = f"approximation_{mode}_{suffix}_{timestamp}.csv"
    save_results_to_csv(results, output_path)

    print(f"\n✅ SELECTION SWEEP COMPLETE: {len([r for r in results if r['success']])}/{len(results)} tests successful")

if __name__ == "__main__":
    asyncio.run(main())
