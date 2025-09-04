#!/usr/bin/env python3
"""
Compare BON-selected outputs with ALL other outputs for each prompt using LLM judge.
For each prompt, compare the selected output against every other output and calculate win percentage.
Includes proper rate limiting and error handling.
"""

import json
import asyncio
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.judges.llm_judge import PersonaJudge
from src.core.attribute_prompts import persona_prompts
import argparse

async def run_comparisons_with_rate_limit(judge, comparisons: List[Dict], 
                                        max_concurrent: int = 10, 
                                        delay_between_batches: float = 1.0):
    """Run comparisons with rate limiting to avoid API limits."""
    results = []
    
    # Process in smaller batches
    batch_size = max_concurrent
    total_batches = (len(comparisons) + batch_size - 1) // batch_size
    
    for i in range(0, len(comparisons), batch_size):
        batch = comparisons[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} comparisons)...")
        
        try:
            # Run batch with retry logic
            batch_results = await run_batch_with_retry(judge, batch)
            results.extend(batch_results)
            
            # Delay between batches to avoid rate limits
            if i + batch_size < len(comparisons):
                print(f"Waiting {delay_between_batches}s before next batch...")
                await asyncio.sleep(delay_between_batches)
                
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            # Add error results for failed batch
            results.extend(["Error"] * len(batch))
    
    return results

async def run_batch_with_retry(judge, batch: List[Dict], max_retries: int = 3):
    """Run a batch with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            return await judge.batch_compare(batch, max_concurrent=len(batch))
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed after {max_retries} retries: {e}")
                raise
            
            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

def main():
    # Load BON data
    data_path = project_root / "data" / "bon_attributes.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Example info - update with your actual drift_bon results
    info = {"user": "user11", "n": 16, "training_size": 150, "lambda": 4e-05, "selected_indices": [5, 5, 0, 5, 5, 5, 13, 5, 7, 10, 5, 5, 5, 14, 7, 5, 7, 3, 10, 10, 10, 5, 13, 5, 10, 2, 5, 5, 5, 5, 5, 10, 10, 5, 5, 3, 5, 13, 7, 5, 5, 10, 13, 5, 10, 7, 10, 10, 10, 10, 10, 13, 2, 13, 3, 3, 9, 5, 10, 10, 13, 10, 10, 13, 5, 10, 5, 2, 7, 13, 2, 5, 13, 10, 13, 5, 7, 13, 5, 13, 7, 15, 5, 5, 7, 15, 13, 10, 7, 10, 13, 10, 11, 10, 10, 10, 12, 13, 5, 13], "num_prompts": 100, "system_prompt_list": "personas"}

    print(f"Comparing BON selections vs ALL other outputs for {info['user']}")
    print(f"N={info['n']}, Lambda={info['lambda']}")
    print(f"Number of prompts: {info['num_prompts']}")
    
    # Initialize judge
    judge = PersonaJudge(base_url="https://api.openai.com/v1", model="gpt-4o")
    
    # Use the same persona as in compare_bon_with_random.py
    persona = None
    with open('data/persona_pref/user_metadata.json', 'r') as f:
        user_metadata = json.load(f)
    
    for user in user_metadata['users']:
        if user['user_id'] == info['user']:
            persona = user['persona_text']
            break
    
    print(f"Using persona: {persona}")
    
    async def compare_all():
        prompt_win_percentages = []
        all_comparisons = []
        prompt_indices_map = []  # Track which comparisons belong to which prompt
        
        # For each prompt, compare selected output against ALL others
        for prompt_idx, selected_index in enumerate(info["selected_indices"]):
            if prompt_idx >= len(data):
                print(f"Warning: Skipping prompt {prompt_idx}, not enough data")
                continue
            
            prompt_data = data[prompt_idx]
            outputs = prompt_data['outputs']
            
            # Check if selected index is valid
            if selected_index >= len(outputs):
                print(f"Warning: Prompt {prompt_idx} - selected index {selected_index} out of range (only {len(outputs)} outputs)")
                continue
            
            selected_output = outputs[selected_index]
            prompt_text = prompt_data['prompt']
            
            print(f"\nPrompt {prompt_idx}: Selected index {selected_index} vs {len(outputs)-1} others")
            
            # Create comparisons for this prompt (selected vs all others)
            prompt_comparisons = []
            for other_idx, other_output in enumerate(outputs):
                if other_idx == selected_index:
                    continue  # Skip comparison with itself
                
                comparison_idx = len(all_comparisons) + len(prompt_comparisons)
                prompt_comparisons.append({
                    'persona': persona,
                    'question': prompt_text,
                    'response_a': selected_output,  # Selected (BON)
                    'response_b': other_output,     # Other output
                    'prompt_idx': prompt_idx,
                    'selected_idx': selected_index,
                    'other_idx': other_idx,
                    'comparison_idx': comparison_idx
                })
                prompt_indices_map.append(prompt_idx)
            
            all_comparisons.extend(prompt_comparisons)
            print(f"  Added {len(prompt_comparisons)} comparisons")
        
        print(f"\nTotal comparisons to run: {len(all_comparisons)}")
        print("This compares each selected output against ALL other outputs for that prompt")
        print("Using conservative rate limiting to avoid API limits...")
        
        # Run all comparisons with conservative rate limiting
        results = await run_comparisons_with_rate_limit(
            judge, 
            all_comparisons, 
            max_concurrent=10,  # Conservative concurrency
            delay_between_batches=4.0  # 2s delay between batches
        )
        
        # Process results by prompt
        prompt_results = {}
        for comp, result in zip(all_comparisons, results):
            prompt_idx = comp['prompt_idx']
            if prompt_idx not in prompt_results:
                prompt_results[prompt_idx] = {
                    'wins': 0,
                    'losses': 0,
                    'ties': 0,
                    'errors': 0,
                    'selected_idx': comp['selected_idx']
                }
            
            if result == "A":  # Selected output wins
                prompt_results[prompt_idx]['wins'] += 1
            elif result == "B":  # Other output wins
                prompt_results[prompt_idx]['losses'] += 1
            elif result == "Error":
                prompt_results[prompt_idx]['errors'] += 1
            else:  # Tie or other
                prompt_results[prompt_idx]['ties'] += 1
        
        # Calculate win percentages
        for prompt_idx in sorted(prompt_results.keys()):
            res = prompt_results[prompt_idx]
            total_valid = res['wins'] + res['losses']
            
            if total_valid > 0:
                win_percentage = res['wins'] / total_valid * 100
            else:
                win_percentage = 0  # Handle case where all comparisons failed
            
            prompt_win_percentages.append(win_percentage)
            
            print(f"Prompt {prompt_idx}: Selected[{res['selected_idx']}] wins {res['wins']}/{total_valid} = {win_percentage:.1f}% (ties: {res['ties']}, errors: {res['errors']})")
        
        return prompt_win_percentages, prompt_results
    
    # Run async comparison
    try:
        win_percentages, prompt_results = asyncio.run(compare_all())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # Calculate statistics
    if win_percentages:
        mean_win_rate = np.mean(win_percentages)
        std_win_rate = np.std(win_percentages)
        min_win_rate = np.min(win_percentages)
        max_win_rate = np.max(win_percentages)
        median_win_rate = np.median(win_percentages)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE RESULTS:")
        print(f"{'='*60}")
        print(f"Number of prompts evaluated: {len(win_percentages)}")
        print(f"")
        print(f"Win Rate Statistics:")
        print(f"  Mean:   {mean_win_rate:.2f}% � {std_win_rate:.2f}%")
        print(f"  Median: {median_win_rate:.2f}%")
        print(f"  Min:    {min_win_rate:.2f}%")
        print(f"  Max:    {max_win_rate:.2f}%")
        print(f"")
        
        # Interpretation
        if mean_win_rate > 50:
            advantage = mean_win_rate - 50
            print(f"<� BON selection shows {advantage:.2f}% advantage over random selection!")
            print(f"   (Random would be ~50%, BON achieves {mean_win_rate:.2f}%)")
        elif mean_win_rate < 50:
            disadvantage = 50 - mean_win_rate
            print(f"�  BON selection shows {disadvantage:.2f}% disadvantage vs random")
            print(f"   (Random would be ~50%, BON achieves {mean_win_rate:.2f}%)")
        else:
            print(f">7 BON selection performs similarly to random selection")
            
        # Distribution analysis
        above_50 = sum(1 for wp in win_percentages if wp > 50)
        above_60 = sum(1 for wp in win_percentages if wp > 60)
        above_70 = sum(1 for wp in win_percentages if wp > 70)
        
        print(f"")
        print(f"Distribution:")
        print(f"  Prompts where BON > 50%: {above_50}/{len(win_percentages)} ({above_50/len(win_percentages)*100:.1f}%)")
        print(f"  Prompts where BON > 60%: {above_60}/{len(win_percentages)} ({above_60/len(win_percentages)*100:.1f}%)")
        print(f"  Prompts where BON > 70%: {above_70}/{len(win_percentages)} ({above_70/len(win_percentages)*100:.1f}%)")
        
        # Show individual results (first 20)
        print(f"")
        print("Individual prompt results (first 20):")
        for i, wp in enumerate(win_percentages[:20]):
            status = "" if wp > 50 else "" if wp < 50 else "="
            print(f"  Prompt {i:3d}: {wp:5.1f}% {status}")
        
        if len(win_percentages) > 20:
            print(f"  ... and {len(win_percentages) - 20} more prompts")
        
        # Save results to file
        results_file = project_root / "results" / "bon_comprehensive_evaluation.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump({
                "info": info,
                "statistics": {
                    "mean_win_rate": mean_win_rate,
                    "std_win_rate": std_win_rate,
                    "median_win_rate": median_win_rate,
                    "min_win_rate": min_win_rate,
                    "max_win_rate": max_win_rate,
                    "num_prompts": len(win_percentages),
                    "prompts_above_50": above_50,
                    "prompts_above_60": above_60,
                    "prompts_above_70": above_70
                },
                "per_prompt_win_percentages": win_percentages,
                "detailed_results": {
                    str(k): v for k, v in prompt_results.items()
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
            
    else:
        print("No valid comparisons completed!")

if __name__ == "__main__":
    main()