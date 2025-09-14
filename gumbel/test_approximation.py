#!/usr/bin/env python3
"""
Test Approximation Script
Uses async utils to compute log probs, finds p vector using drift approximation logic,
then evaluates preference pair accuracy on test data.
"""

import asyncio
import json
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import argparse
from transformers import AutoTokenizer

# Import async utils for log prob computation
from async_utils import fetch_sum_lp, build_full_prompt, VLLM_URL, MODEL_ID, CONCURRENCY
import aiohttp

# Import drift approximation logic
import sys
import os
sys.path.append('../utils')
from attribute_prompts import attribute_prompts, base_prompt


def l1_solve(d_mean, l1_lambda, std=None):
    """
    Closed-form solution to: maximize d^T p - lambda * ||p||_1  s.t. ||p||_2 <= 1
    Copied from drift.py
    """
    d = np.asarray(d_mean, dtype=float)
    # soft-threshold
    z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
    norm = np.linalg.norm(z, ord=2)
    if norm == 0.0:
        return np.zeros_like(d)
    if std is None:
        return z / norm
    else:
        return z / (norm * std)


async def get_log_probs_async(session: aiohttp.ClientSession, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str]) -> Tuple[List[float], List[int]]:
    """
    Async version of get_log_probs using aiohttp
    """
    tasks = []
    prompts_data = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        full_prompt, n_prefix, comp_len = build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
        prompts_data.append((n_prefix, comp_len))
        tasks.append(fetch_sum_lp(session, full_prompt, n_prefix, comp_len))
    
    log_probs = await asyncio.gather(*tasks)
    token_counts = [comp_len for _, comp_len in prompts_data]
    
    return log_probs, token_counts


async def approximate_async(data: List[Tuple[str, str, str]], tokenizer, s0: str, s_list: List[str], l1_lambda: float = 0.01) -> np.ndarray:
    """
    Async version of the approximate function from drift.py
    
    Args:
        data: list of (question, y_w, y_l) tuples
        tokenizer: tokenizer
        s0: base system prompt
        s_list: list of attribute system prompts
        l1_lambda: L1 regularization parameter
    
    Returns:
        p vector (numpy array)
    """
    m, k = len(data), len(s_list)
    questions, yw_list, yl_list = zip(*data)
    
    # Set up async session
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Compute base probabilities
        print("Computing base probabilities...")
        pi_yw_base, cnt_yw_base = await get_log_probs_async(session, tokenizer, [s0]*m, questions, yw_list)
        pi_yl_base, cnt_yl_base = await get_log_probs_async(session, tokenizer, [s0]*m, questions, yl_list)
        
        # Convert to tensors
        pi_yw_base = torch.tensor(pi_yw_base, dtype=torch.float32)
        cnt_yw_base = torch.tensor(cnt_yw_base, dtype=torch.float32)
        pi_yl_base = torch.tensor(pi_yl_base, dtype=torch.float32)
        cnt_yl_base = torch.tensor(cnt_yl_base, dtype=torch.float32)
        
        # Safe average log-probs
        eps = 1e-12
        yw_base_avg = pi_yw_base / torch.clamp(cnt_yw_base, min=eps)
        yl_base_avg = pi_yl_base / torch.clamp(cnt_yl_base, min=eps)
        
        # Build X matrix
        X = torch.zeros((m, k), dtype=torch.float32)
        
        for j, system in enumerate(s_list):
            print(f"Processing attribute {j+1}/{k}: {system[:50]}...")
            
            pi_yw_attr, cnt_yw_attr = await get_log_probs_async(session, tokenizer, [system]*m, questions, yw_list)
            pi_yl_attr, cnt_yl_attr = await get_log_probs_async(session, tokenizer, [system]*m, questions, yl_list)
            
            pi_yw_attr = torch.tensor(pi_yw_attr, dtype=torch.float32)
            cnt_yw_attr = torch.tensor(cnt_yw_attr, dtype=torch.float32)
            pi_yl_attr = torch.tensor(pi_yl_attr, dtype=torch.float32)
            cnt_yl_attr = torch.tensor(cnt_yl_attr, dtype=torch.float32)
            
            yw_attr_avg = pi_yw_attr / torch.clamp(cnt_yw_attr, min=eps)
            yl_attr_avg = pi_yl_attr / torch.clamp(cnt_yl_attr, min=eps)
            
            # Column j: (yw_attr - yw_base) - (yl_attr - yl_base)
            X[:, j] = (yw_attr_avg - yw_base_avg) - (yl_attr_avg - yl_base_avg)
    
    # Compute drift direction
    col_std = X.std(dim=0).clamp_min(1e-8)
    d = (X / col_std).mean(dim=0).detach().cpu().numpy()
    
    # Solve for p vector
    p = l1_solve(d, l1_lambda, std=col_std.detach().cpu().numpy())
    
    return p


async def evaluate_accuracy_async(test_data: List[Dict[str, str]], p: np.ndarray, tokenizer, base_prompt: str, attribute_prompts: List[str]) -> float:
    """
    Evaluate preference pair accuracy on test data using the learned p vector
    
    Args:
        test_data: list of preference pairs with 'prompt', 'chosen', 'rejected'
        p: learned drift vector
        tokenizer: tokenizer
        base_prompt: base system prompt
        attribute_prompts: list of attribute prompts
    
    Returns:
        accuracy (float)
    """
    n = len(test_data)
    prompts = [item['prompt'] for item in test_data]
    chosen = [item['chosen'] for item in test_data]
    rejected = [item['rejected'] for item in test_data]
    
    # Set up async session
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Get base log probabilities
        print("Computing base log probabilities for test data...")
        chosen_base_probs, chosen_base_counts = await get_log_probs_async(session, tokenizer, [base_prompt]*n, prompts, chosen)
        rejected_base_probs, rejected_base_counts = await get_log_probs_async(session, tokenizer, [base_prompt]*n, prompts, rejected)
        
        # Initialize drift scores
        drift_scores = torch.zeros(n, dtype=torch.float32)
        
        # Process each attribute
        for i, attr_prompt in enumerate(attribute_prompts):
            if p[i] == 0:
                continue
                
            print(f"Processing test attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
            
            chosen_attr_probs, chosen_attr_counts = await get_log_probs_async(session, tokenizer, [attr_prompt]*n, prompts, chosen)
            rejected_attr_probs, rejected_attr_counts = await get_log_probs_async(session, tokenizer, [attr_prompt]*n, prompts, rejected)
            
            # Convert to tensors and compute averages
            chosen_attr_avg = torch.tensor(chosen_attr_probs, dtype=torch.float32) / torch.tensor(chosen_attr_counts, dtype=torch.float32)
            rejected_attr_avg = torch.tensor(rejected_attr_probs, dtype=torch.float32) / torch.tensor(rejected_attr_counts, dtype=torch.float32)
            chosen_base_avg = torch.tensor(chosen_base_probs, dtype=torch.float32) / torch.tensor(chosen_base_counts, dtype=torch.float32)
            rejected_base_avg = torch.tensor(rejected_base_probs, dtype=torch.float32) / torch.tensor(rejected_base_counts, dtype=torch.float32)
            
            # Compute drift contribution: p[i] * ((chosen_attr - chosen_base) - (rejected_attr - rejected_base))
            attribute_drift = p[i] * ((chosen_attr_avg - chosen_base_avg) - (rejected_attr_avg - rejected_base_avg))
            drift_scores += attribute_drift
    
    # Count correct predictions (positive drift means chosen > rejected)
    correct = (drift_scores > 0).sum().item()
    accuracy = correct / n
    
    return accuracy


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