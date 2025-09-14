#!/usr/bin/env python3
"""
Compare BON-selected outputs with randomly selected outputs using LLM judge.
Fixed version with proper rate limiting and error handling.
"""

import json
import random
import asyncio
import sys
import os
from pathlib import Path
import time
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.judges.llm_judge import PersonaJudge

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
    
    info1 = {"user": "user20", "n": 16, "training_size": 150, "lambda": 0.0001, "selected_indices": [5, 5, 0, 5, 5, 5, 13, 5, 7, 13, 5, 5, 13, 12, 3, 5, 7, 3, 10, 10, 5, 5, 13, 13, 10, 3, 5, 14, 2, 5, 5, 10, 7, 5, 5, 15, 5, 13, 7, 5, 13, 5, 13, 5, 13, 7, 10, 10, 13, 5, 3, 13, 13, 13, 5, 3, 13, 5, 13, 5, 13, 10, 13, 13, 5, 5, 5, 2, 5, 13, 2, 5, 13, 10, 13, 15, 7, 13, 13, 13, 7, 15, 5, 5, 7, 15, 13, 10, 7, 13, 13, 10, 11, 10, 13, 10, 14, 13, 5, 13], "num_prompts": 100, "system_prompt_list": "personas"}
    info2 = {"user": "user20", "n": 16, "training_size": 150, "selected_indices": [2, 13, 2, 2, 9, 8, 8, 8, 13, 11, 13, 0, 8, 13, 12, 2, 2, 1, 4, 2, 6, 8, 14, 9, 13, 14, 13, 8, 13, 14, 13, 11, 14, 8, 0, 13, 11, 14, 11, 13, 14, 11, 13, 13, 8, 8, 6, 10, 9, 6, 13, 8, 14, 3, 7, 8, 2, 11, 14, 0, 12, 0, 0, 14, 14, 13, 3, 8, 0, 9, 7, 14, 14, 13, 14, 14, 13, 13, 14, 13, 4, 13, 6, 10, 1, 1, 0, 2, 2, 1, 14, 8, 14, 13, 8, 0, 8, 8, 13, 13]}

    print(f"Comparing BON selections: {info1['user']} vs {info2['user']}")
    print(f"User1 N={info1['n']}, User2 N={info2['n']}")
    print(f"Training sizes: {info1['training_size']} vs {info2['training_size']}")
    
    # Initialize judge once
    judge = PersonaJudge(base_url="https://api.openai.com/v1", model="gpt-4o")
    
    # Track wins
    user1_wins = 0
    user2_wins = 0
    ties = 0
    errors = 0

    # Get personas for both users
    persona = None
    with open('data/persona_pref/user_metadata.json', 'r') as f:
        user_metadata = json.load(f)
    
    for user in user_metadata['users']:
        if user['user_id'] == info1['user']:
            persona = user['persona_text']
    
    if persona is None:
        print("Error: Could not find personas for one or both users")
        return
    
    async def compare_all():
        nonlocal user1_wins, user2_wins, ties, errors
        
        comparisons = []
        
        # Prepare all comparisons
        for i, (selected_index1, selected_index2) in enumerate(zip(info1["selected_indices"], info2["selected_indices"])):
            if i >= len(data):
                print(f"Warning: Skipping index {i}, not enough data")
                continue
            
            # Get outputs
            try:
                user1_output = data[i]['outputs'][selected_index1]
                user2_output = data[i]['outputs'][selected_index2]
            except IndexError as e:
                print(f"Prompt {i}: Index error - {e}, skipping")
                errors += 1
                continue
            
            # Get prompt
            prompt = data[i]['prompt']
            
            comparisons.append({
                'persona': persona,  # Use persona1 for the judge
                'question': prompt,
                'response_a': user1_output,  # User1 selected
                'response_b': user2_output,  # User2 selected
                'prompt_idx': i,
                'user1_idx': selected_index1,
                'user2_idx': selected_index2
            })
        
        print(f"Running {len(comparisons)} comparisons...")
        print("Using conservative rate limiting to avoid API limits...")
        
        # Run comparisons with rate limiting
        # Very conservative settings: 2 concurrent, 2s delay between batches
        results = await run_comparisons_with_rate_limit(
            judge, comparisons, 
            max_concurrent=10,  # Lower concurrency
            delay_between_batches=2.0  # Longer delay
        )
        
        # Process results
        for comp, result in zip(comparisons, results):
            i = comp['prompt_idx']
            
            if result == "A":
                user1_wins += 1
                winner = f"{info1['user']}"
            elif result == "B":
                user2_wins += 1
                winner = f"{info2['user']}"
            elif result == "Error":
                errors += 1
                winner = "Error"
            else:
                ties += 1
                winner = "Tie"
            
            print(f"Prompt {i}: {info1['user']}[{comp['user1_idx']}] vs {info2['user']}[{comp['user2_idx']}] -> {winner}")
    
    # Run async comparison
    try:
        asyncio.run(compare_all())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return
    
    # Print results
    total_comparisons = user1_wins + user2_wins + ties + errors
    valid_comparisons = user1_wins + user2_wins + ties
    
    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"{'='*50}")
    print(f"{info1['user']} wins: {user1_wins} ({user1_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"{info2['user']} wins: {user2_wins} ({user2_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"Ties: {ties} ({ties/valid_comparisons*100:.1f}% of valid)")
    print(f"Errors: {errors}")
    print(f"Total attempted: {total_comparisons}")
    print(f"Valid comparisons: {valid_comparisons}")
    
    if valid_comparisons > 0:
        if user1_wins > user2_wins:
            print(f"\n🎉 {info1['user']} outperforms {info2['user']} by {user1_wins - user2_wins} wins!")
        elif user2_wins > user1_wins:
            print(f"\n🎉 {info2['user']} outperforms {info1['user']} by {user2_wins - user1_wins} wins")
        else:
            print(f"\n🤷 {info1['user']} and {info2['user']} perform equally")
    else:
        print(f"\n❌ No valid comparisons completed")
    
    # Save results to file
    results_file = project_root / "results" / "bon_evaluation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump({
            "info1": info1,
            "info2": info2,
            "results": {
                "user1_wins": user1_wins,
                "user2_wins": user2_wins,
                "ties": ties,
                "errors": errors,
                "total_attempted": total_comparisons,
                "valid_comparisons": valid_comparisons
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()