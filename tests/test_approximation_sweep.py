#!/usr/bin/env python3
"""
Approximation Sweep by Number of Features

- Loads a JSON of feature sets like:
  [
    {"sparsity": 0.0001, "lr": 0.01, "features": [2,3,4,...]},
    {"sparsity": 0.001,  "lr": 0.01, "features": [3,13,27,...]},
    ...
  ]
  (Only 'features' is required; 'sparsity'/'lr' are ignored.)

- Sweeps L1 lambda values for each requested feature count (--features N1 N2 ...).
- Reports/exports results keyed by 'num_features'.
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

from gumbel.utils.async_utils import approximate_async, evaluate_accuracy_async, MODEL_ID

# Add configs directory to path
sys.path.insert(0, '../configs')

def load_attribute_prompts_from_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data["prompts"]

def build_feature_map(features_json_path):
    """
    Returns: dict {num_features: [indices...]}
    If multiple entries share the same count, the first is used.
    """
    with open(features_json_path, 'r') as f:
        data = json.load(f)
    feature_map = {}
    for entry in data:
        feats = entry.get("features", [])
        if not isinstance(feats, list):
            continue
        n = len(feats)
        if n not in feature_map:
            feature_map[n] = feats
    return feature_map

async def run_single_test(train_data, test_data, tokenizer, base_prompt,
                          selected_attributes, l1_lambda, num_features):
    print(f"\n🔧 TESTING: num_features={num_features}, L1={l1_lambda}, Attributes={len(selected_attributes)}")
    try:
        p = await approximate_async(train_data, tokenizer, base_prompt, selected_attributes, l1_lambda)
        accuracy = await evaluate_accuracy_async(test_data, p, tokenizer, base_prompt, selected_attributes)

        non_zero_components = int(np.count_nonzero(p))
        p_norm = float(np.linalg.norm(p))
        sparsity_ratio = (non_zero_components / float(len(p))) if len(p) > 0 else 0.0

        # Top attributes (by |weight|)
        idx_sorted = np.argsort(np.abs(p))
        top_idx = idx_sorted[-5:][::-1]
        top_weights = [float(p[i]) for i in top_idx]
        top_attributes = []
        for i in top_idx:
            s = selected_attributes[i]
            if len(s) > 50:
                s = s[:50] + "..."
            top_attributes.append(s)

        result = {
            'num_features': int(num_features),
            'l1_lambda': float(l1_lambda),
            'num_attributes': len(selected_attributes),
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
            'num_features': int(num_features),
            'l1_lambda': float(l1_lambda),
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
    if not results:
        print("⚠️  No results to save")
        return

    has_user = any('user' in r for r in results)
    fieldnames = [
        'num_features', 'l1_lambda', 'num_attributes', 'accuracy',
        'non_zero_components', 'sparsity_ratio', 'p_norm', 'success', 'error',
        'top_weight_1', 'top_weight_2', 'top_weight_3', 'top_weight_4', 'top_weight_5',
        'top_attr_1', 'top_attr_2', 'top_attr_3', 'top_attr_4', 'top_attr_5'
    ]
    if has_user:
        fieldnames.insert(0, 'user')

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: v for k, v in r.items() if k not in ['top_weights', 'top_attributes']}
            tw = list(r.get('top_weights', []))[:5] + [None] * (5 - len(r.get('top_weights', [])))
            ta = list(r.get('top_attributes', []))[:5] + [None] * (5 - len(r.get('top_attributes', [])))
            for i in range(5):
                row[f'top_weight_{i+1}'] = tw[i]
                row[f'top_attr_{i+1}'] = ta[i]
            writer.writerow(row)

    print(f"💾 Results saved to: {output_path}")

def print_summary_table(results):
    if not results:
        print("⚠️  No results to summarize")
        return

    has_user = any('user' in r for r in results)

    print("\n" + "="*90)
    print("SUMMARY RESULTS")
    print("="*90)

    if has_user:
        print(f"{'User':<8} {'NumFeat':<10} {'L1 Lambda':<10} {'Attributes':<12} "
              f"{'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 90)
        for r in results:
            if r['success']:
                print(f"{r.get('user','N/A'):<8} {r['num_features']:<10} {r['l1_lambda']:<10.3f} {r['num_attributes']:<12} "
                      f"{r['accuracy']:<10.4f} {r['non_zero_components']:<10} {r['p_norm']:<10.4f}")
            else:
                print(f"{r.get('user','N/A'):<8} {r['num_features']:<10} {r['l1_lambda']:<10.3f} {r['num_attributes']:<12} "
                      f"{'FAILED':<10} {'N/A':<10} {'N/A':<10}")
    else:
        print(f"{'NumFeat':<10} {'L1 Lambda':<10} {'Attributes':<12} {'Accuracy':<10} {'Non-Zero':<10} {'P Norm':<10}")
        print("-" * 80)
        for r in results:
            if r['success']:
                print(f"{r['num_features']:<10} {r['l1_lambda']:<10.3f} {r['num_attributes']:<12} "
                      f"{r['accuracy']:<10.4f} {r['non_zero_components']:<10} {r['p_norm']:<10.4f}")
            else:
                print(f"{r['num_features']:<10} {r['l1_lambda']:<10.3f} {r['num_attributes']:<12} "
                      f"{'FAILED':<10} {'N/A':<10} {'N/A':<10}")

    ok = [r for r in results if r['success']]
    if ok:
        best = max(ok, key=lambda x: x['accuracy'])
        if has_user:
            print(f"\n🏆 BEST ACCURACY: {best['accuracy']:.4f} "
                  f"(User: {best.get('user','N/A')}, NumFeat: {best['num_features']}, L1: {best['l1_lambda']})")
        else:
            print(f"\n🏆 BEST ACCURACY: {best['accuracy']:.4f} "
                  f"(NumFeat: {best['num_features']}, L1: {best['l1_lambda']})")

def parse_user_range(user_range_str):
    if not user_range_str:
        return []
    users = []
    for part in user_range_str.split(','):
        part = part.strip()
        if '-' in part and part.startswith('user'):
            prefix = 'user'
            rng = part[len(prefix):]
            if '-' in rng:
                a, b = rng.split('-')
                try:
                    a, b = int(a), int(b)
                    for i in range(a, b + 1):
                        users.append(f"{prefix}{i}")
                except ValueError:
                    users.append(part)
            else:
                users.append(part)
        else:
            users.append(part)
    return users

async def run_user_sweep(user_name, data_dir, tokenizer, base_prompt,
                         available_prompts, args, feature_map):
    print(f"\n👤 PROCESSING USER: {user_name}")
    train_path = os.path.join(data_dir, f"{user_name}_train.json")
    test_path  = os.path.join(data_dir, f"{user_name}_test.json")

    if not os.path.exists(train_path):
        print(f"❌ Training file not found: {train_path}")
        return []
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return []

    # Load train
    with open(train_path, 'r') as f:
        train_raw = json.load(f)
    train_data = [(it['prompt'], it['chosen'], it['rejected'])
                  for it in train_raw[:args.max_train_samples]]
    print(f"Loaded {len(train_data)} training samples")

    # Load test
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    # Sweep
    total_tests = len(args.features) * len(args.lambda_values)
    current = 0
    user_results = []

    for nf in args.features:
        if nf not in feature_map:
            print(f"  ⚠️  No feature set of size {nf} in {args.features_json}; skipping")
            continue
        idxs = feature_map[nf]
        selected_attributes = [available_prompts[i] for i in idxs if 0 <= i < len(available_prompts)]
        for l1 in args.lambda_values:
            current += 1
            print(f"\n  [{current}/{total_tests}] {user_name}", end=" ")
            res = await run_single_test(
                train_data, test_data, tokenizer, base_prompt,
                selected_attributes, l1, nf
            )
            res['user'] = user_name
            user_results.append(res)

    return user_results

async def main():
    parser = argparse.ArgumentParser(description="Approximation sweep by number of features (from JSON feature sets)")

    # Single user
    parser.add_argument("--train-data", type=str, help="Training data path (single user mode)")
    parser.add_argument("--test-data",  type=str, help="Test data path (single user mode)")

    # Multi-user
    parser.add_argument("--data-dir", type=str, default="data/persona_pref/", help="Directory containing user data files")
    parser.add_argument("--users",    type=str, help="User range (e.g., 'user1-5' or 'user1,user3,user5')")

    # Common
    parser.add_argument("--max-train-samples", type=int, default=150, help="Max training samples")
    parser.add_argument("--lambda-values", type=float, nargs='+', default=[0.01], help="L1 lambda values to test")
    parser.add_argument("--features", type=int, nargs='+', default=[5, 23, 194], help="Feature counts to test (match counts present in JSON)")
    parser.add_argument("--features-json", type=str, default="results/gumbel_parameters.json", help="JSON file with [{'features':[...]}] entries")
    parser.add_argument("--output", type=str, help="Output CSV path (optional)")
    parser.add_argument("--attribute-config", type=str, default="gumbel/configs/attribute_prompts.json", help="Path to attribute prompts JSON")

    args = parser.parse_args()

    # Tokenizer & prompts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    base_prompt = "You are a helpful assistant."
    print(f"\n📝 Loading attribute prompts from {args.attribute_config}")
    available_prompts = load_attribute_prompts_from_config(args.attribute_config)
    print(f"Loaded {len(available_prompts)} total attribute prompts")

    # Feature sets
    feature_map = build_feature_map(args.features_json)
    print(f"📦 Loaded {len(feature_map)} unique feature-set sizes from {args.features_json}")

    # Mode detection
    if args.users:
        user_list = parse_user_range(args.users)
        print(f"\n🚀 MULTI-USER SWEEP")
        print(f"Users: {user_list}")
        print(f"Feature sizes requested: {args.features}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Max training samples per user: {args.max_train_samples}")

        all_results = []
        for user in user_list:
            user_results = await run_user_sweep(user, args.data_dir, tokenizer, base_prompt,
                                                available_prompts, args, feature_map)
            all_results.extend(user_results)
        results = all_results
    else:
        # Single user
        if not args.train_data or not args.test_data:
            print("❌ Error: For single user mode, both --train-data and --test-data are required")
            return
        print(f"\n🚀 SINGLE-USER SWEEP")
        print(f"Training data: {args.train_data}")
        print(f"Test data: {args.test_data}")
        print(f"Feature sizes requested: {args.features}")
        print(f"L1 lambda values: {args.lambda_values}")
        print(f"Max training samples: {args.max_train_samples}")

        with open(args.train_data, 'r') as f:
            train_raw = json.load(f)
        train_data = [(it['prompt'], it['chosen'], it['rejected'])
                      for it in train_raw[:args.max_train_samples]]
        print(f"Loaded {len(train_data)} training samples")

        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")

        results = []
        total = len(args.features) * len(args.lambda_values)
        current = 0
        for nf in args.features:
            if nf not in feature_map:
                print(f"⚠️  No feature set of size {nf} in {args.features_json}; skipping")
                continue
            idxs = feature_map[nf]
            selected_attributes = [available_prompts[i] for i in idxs if 0 <= i < len(available_prompts)]
            for l1 in args.lambda_values:
                current += 1
                print(f"\n[{current}/{total}]", end=" ")
                res = await run_single_test(train_data, test_data, tokenizer, base_prompt,
                                            selected_attributes, l1, nf)
                results.append(res)

    # Summary + save
    print_summary_table(results)
    if args.output:
        out = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "multi" if args.users else "single"
        out = f"approximation_sweep_{suffix}_features_{timestamp}.csv"
    save_results_to_csv(results, out)

    ok = [r for r in results if r['success']]
    print(f"\n✅ SWEEP COMPLETE: {len(ok)}/{len(results)} tests successful")

if __name__ == "__main__":
    asyncio.run(main())
