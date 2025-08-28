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
    
    info = {"user": "user8", "n": 16, "training_size": 200, "lambda": 0.01, "selected_indices": [5, 5, 2, 5, 5, 5, 5, 5, 7, 13, 5, 5, 5, 7, 0, 5, 5, 5, 7, 5, 5, 5, 5, 5, 9, 13, 5, 2, 5, 5, 5, 10, 12, 5, 5, 7, 5, 5, 7, 5, 5, 5, 5, 5, 3, 5, 7, 5, 13, 5, 5, 5, 2, 13, 5, 3, 7, 5, 7, 5, 7, 13, 5, 10, 5, 4, 5, 5, 5, 5, 9, 5, 5, 3, 4, 5, 7, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 7, 13, 13, 5, 10, 5, 7, 13, 5, 7, 13, 5], "num_prompts": 100}

    print(f"Comparing BON selections vs random for {info['user']}")
    print(f"N={info['n']}, Lambda={info['lambda']}")
    print(f"Number of prompts: {info['num_prompts']}")
    
    # Initialize judge once
    judge = PersonaJudge(base_url="https://api.openai.com/v1", model="gpt-4o")
    
    # Track wins
    bon_wins = 0
    random_wins = 0
    ties = 0
    errors = 0
    
    # Set random seed for reproducibility
    random.seed(42)
    
    async def compare_all():
        nonlocal bon_wins, random_wins, ties, errors
        
        comparisons = []
        
        # Prepare all comparisons
        for i, selected_index in enumerate(info["selected_indices"]):
            if i >= len(data):
                print(f"Warning: Skipping index {i}, not enough data")
                continue
            
            # Get random index (make sure it's valid and different from BON selection)
            max_outputs = len(data[i]['outputs'])
            random_index = random.randint(0, max_outputs - 1)
            
            # Retry if same as BON selection (up to 5 times)
            retries = 0
            while random_index == selected_index and retries < 5:
                random_index = random.randint(0, max_outputs - 1)
                retries += 1
            
            # Skip if they're still the same after retries
            if random_index == selected_index:
                print(f"Prompt {i}: Could not find different random index, skipping")
                ties += 1
                continue
            
            # Get outputs
            try:
                random_output = data[i]['outputs'][random_index]
                bon_output = data[i]['outputs'][selected_index]
            except IndexError as e:
                print(f"Prompt {i}: Index error - {e}, skipping")
                errors += 1
                continue
            
            # Get prompt and persona
            prompt = data[i]['prompt']
            persona = "You are an AI assistant who speaks like a seasoned comic. You are playful, often irreverent. You tend to respond with clever jokes, sarcasm, and punchy comebacks. You value humor, levity, and not taking things too seriously."
            
            comparisons.append({
                'persona': persona,
                'question': prompt,
                'response_a': bon_output,  # BON selected
                'response_b': random_output,  # Random
                'prompt_idx': i,
                'bon_idx': selected_index,
                'random_idx': random_index
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
                bon_wins += 1
                winner = "BON"
            elif result == "B":
                random_wins += 1
                winner = "Random"
            elif result == "Error":
                errors += 1
                winner = "Error"
            else:
                ties += 1
                winner = "Tie"
            
            print(f"Prompt {i}: BON[{comp['bon_idx']}] vs Random[{comp['random_idx']}] -> {winner}")
    
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
    total_comparisons = bon_wins + random_wins + ties + errors
    valid_comparisons = bon_wins + random_wins + ties
    
    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"{'='*50}")
    print(f"BON wins: {bon_wins} ({bon_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"Random wins: {random_wins} ({random_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"Ties: {ties} ({ties/valid_comparisons*100:.1f}% of valid)")
    print(f"Errors: {errors}")
    print(f"Total attempted: {total_comparisons}")
    print(f"Valid comparisons: {valid_comparisons}")
    
    if valid_comparisons > 0:
        if bon_wins > random_wins:
            print(f"\n🎉 BON selection outperforms random by {bon_wins - random_wins} wins!")
        elif random_wins > bon_wins:
            print(f"\n⚠️  Random selection outperforms BON by {random_wins - bon_wins} wins")
        else:
            print(f"\n🤷 BON and random perform equally")
    else:
        print(f"\n❌ No valid comparisons completed")
    
    # Save results to file
    results_file = project_root / "results" / "bon_evaluation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump({
            "info": info,
            "results": {
                "bon_wins": bon_wins,
                "random_wins": random_wins,
                "ties": ties,
                "errors": errors,
                "total_attempted": total_comparisons,
                "valid_comparisons": valid_comparisons
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()