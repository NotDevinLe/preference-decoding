#!/usr/bin/env python3
"""
Prepare evaluation data from source datasets (e.g., Dolly).
Creates a standardized evaluation prompt dataset.

Usage:
    python scripts/generate/prepare_evaluation_data.py \
        --config configs/experiment_config.yaml \
        --output data/processed/evaluation_prompts.json
"""

import os
import sys
import json
import yaml
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dolly_dataset(num_prompts: int = 1000) -> List[Dict[str, str]]:
    """
    Load Dolly dataset and convert to evaluation format.
    
    Args:
        num_prompts: Number of prompts to sample
        
    Returns:
        List of prompt dictionaries
    """
    print("Loading Dolly dataset...")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    prompts = []
    categories_seen = set()
    
    for i, item in enumerate(tqdm(dataset)):
        # Extract prompt and context
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        category = item.get('category', 'general')
        
        # Combine instruction and context if both exist
        if context:
            prompt = f"Context: {context}\n\nQuestion: {instruction}"
        else:
            prompt = instruction
        
        # Skip empty prompts
        if not prompt.strip():
            continue
        
        # Create evaluation entry
        entry = {
            'id': i,
            'prompt': prompt.strip(),
            'category': category,
            'source': 'dolly'
        }
        
        prompts.append(entry)
        categories_seen.add(category)
        
        if len(prompts) >= num_prompts:
            break
    
    print(f"Loaded {len(prompts)} prompts from {len(categories_seen)} categories")
    print(f"Categories: {sorted(categories_seen)}")
    
    return prompts




def filter_and_balance_prompts(
    prompts: List[Dict],
    target_count: int,
    balance_by: str = 'category'
) -> List[Dict]:
    """
    Filter and balance prompts by category/complexity/etc.
    
    Args:
        prompts: List of prompts
        target_count: Target number of prompts
        balance_by: Field to balance by
        
    Returns:
        Balanced subset of prompts
    """
    print(f"Balancing {len(prompts)} prompts to {target_count} by {balance_by}")
    
    # Group by balance field
    groups = {}
    for prompt in prompts:
        key = prompt.get(balance_by, 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(prompt)
    
    # Calculate target per group
    num_groups = len(groups)
    prompts_per_group = target_count // num_groups
    remainder = target_count % num_groups
    
    balanced_prompts = []
    group_names = sorted(groups.keys())
    
    for i, group_name in enumerate(group_names):
        group_prompts = groups[group_name]
        
        # Add extra prompt to first 'remainder' groups
        group_target = prompts_per_group + (1 if i < remainder else 0)
        
        # Sample from this group
        if len(group_prompts) >= group_target:
            selected = random.sample(group_prompts, group_target)
        else:
            selected = group_prompts  # Take all if not enough
        
        balanced_prompts.extend(selected)
        print(f"  {group_name}: {len(selected)} prompts")
    
    random.shuffle(balanced_prompts)  # Final shuffle
    return balanced_prompts[:target_count]


def add_evaluation_metadata(prompts: List[Dict]) -> List[Dict]:
    """
    Add metadata for evaluation purposes.
    
    Args:
        prompts: List of prompts
        
    Returns:
        Prompts with added metadata
    """
    for i, prompt in enumerate(prompts):
        # Add evaluation ID
        prompt['eval_id'] = i
        
        # Estimate complexity based on prompt length and question type
        prompt_text = prompt['prompt']
        word_count = len(prompt_text.split())
        
        if word_count < 20:
            complexity = 'simple'
        elif word_count < 50:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        prompt['complexity'] = complexity
        
        
        # Add expected response characteristics
        prompt['expected_length'] = 'medium'  # Can be refined later
        prompt['requires_reasoning'] = any(word in prompt_text.lower() 
                                         for word in ['why', 'how', 'explain', 'analyze'])
        prompt['requires_creativity'] = any(word in prompt_text.lower()
                                          for word in ['create', 'imagine', 'write', 'design'])
    
    return prompts




def validate_prompts(prompts: List[Dict]) -> List[Dict]:
    """
    Validate and clean prompts.
    
    Args:
        prompts: List of prompts to validate
        
    Returns:
        Cleaned prompts
    """
    valid_prompts = []
    
    for prompt in prompts:
        # Check required fields
        if not all(key in prompt for key in ['id', 'prompt']):
            continue
        
        # Check prompt quality
        if len(prompt['prompt'].strip()) < 10:
            continue
        
        # Check for duplicates (simple check on first 50 chars)
        prompt_start = prompt['prompt'][:50]
        if any(p['prompt'][:50] == prompt_start for p in valid_prompts):
            continue
        
        valid_prompts.append(prompt)
    
    print(f"Validated {len(valid_prompts)} prompts (removed {len(prompts) - len(valid_prompts)})")
    return valid_prompts


def save_prompts(prompts: List[Dict], output_path: str):
    """Save prompts to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create final structure
    data = {
        'metadata': {
            'num_prompts': len(prompts),
            'categories': list(set(p.get('category', 'unknown') for p in prompts)),
        },
        'prompts': prompts
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(prompts)} prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation data from source datasets"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment configuration"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/evaluation_prompts.json",
        help="Output path for evaluation prompts"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="dolly",
        choices=["dolly", "custom"],
        help="Source dataset to use"
    )
    
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of prompts to generate (overrides config)"
    )
    
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    num_prompts = args.num_prompts or config['data']['num_evaluation_prompts']
    
    print(f"Preparing {num_prompts} evaluation prompts...")
    
    # Load base prompts
    if args.source == "dolly":
        prompts = load_dolly_dataset(num_prompts * 2)  # Load extra for filtering
    else:
        raise ValueError(f"Unknown source: {args.source}")
    
    
    # Balance and filter
    prompts = filter_and_balance_prompts(prompts, num_prompts, 'category')
    
    # Add basic eval IDs and validate
    valid_prompts = []
    for i, prompt in enumerate(prompts):
        if prompt.get('prompt', '').strip() and len(prompt['prompt'].strip()) >= 10:
            prompt['eval_id'] = i
            valid_prompts.append(prompt)
    
    prompts = valid_prompts
    print(f"Validated {len(prompts)} prompts")
    
    # Save
    save_prompts(prompts, args.output)
    
    # Print summary
    print("\n=== EVALUATION DATA SUMMARY ===")
    categories = {}
    
    for prompt in prompts:
        cat = prompt.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"Total prompts: {len(prompts)}")
    print(f"Categories: {dict(sorted(categories.items()))}")
    
    print(f"\nData saved to: {args.output}")


if __name__ == "__main__":
    main()