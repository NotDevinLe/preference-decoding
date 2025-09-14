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

    user = "user18"
    with open(data_path, "r") as f:
        data = json.load(f)
    
    print(f"Comparing Compressed vs ICL")

    
    # Initialize judge once
    judge = PersonaJudge(base_url="https://api.openai.com/v1", model="gpt-4o")
    
    # Track wins
    compress_wins = 0
    icl_wins = 0
    ties = 0
    errors = 0

    persona = ""

    with open('data/persona_pref/user_metadata.json', 'r') as f:
        user_metadata = json.load(f)
    
    with open(f'results/icl_gen_{user}_open_oneshot.json', 'r') as f:
        icl_data = json.load(f)
    
    with open(f'results/compressed_responses/{user}.json', 'r') as f:
        compress_data = json.load(f)
    
    for user in user_metadata['users']:
        if user['user_id'] == user:
            persona = user['persona_text']
            break
    
    # Set random seed for reproducibility
    random.seed(42)
    
    async def compare_all():
        nonlocal compress_wins, icl_wins, ties, errors
        
        comparisons = []
        
        # Prepare all comparisons
        for i, selected_index in enumerate(compress_data):
            # Get outputs
            try:
                compress_output = compress_data[i]
                icl_output = icl_data[i]
            except IndexError as e:
                print(f"Prompt {i}: Index error - {e}, skipping")
                errors += 1
                continue
            
            # Get prompt and persona
            prompt = data[i]['prompt']
            
            comparisons.append({
                'persona': persona,
                'question': prompt,
                'response_a': icl_output,
                'response_b': compress_output,
                'prompt_idx': i,
                'compress_idx': selected_index,
                'icl_idx': i
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
                icl_wins += 1
                winner = "ICL"
            elif result == "B":
                compress_wins += 1
                winner = "COMPRESS"
            elif result == "Error":
                errors += 1
                winner = "Error"
            else:
                ties += 1
                winner = "Tie"
            
            print(f"Prompt {i}: [{comp['icl_idx']}] vs [{comp['compress_idx']}] -> {winner}")
    
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
    total_comparisons = compress_wins + icl_wins + ties + errors
    valid_comparisons = compress_wins + icl_wins + ties
    
    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"{'='*50}")
    print(f"COMPRESS wins: {compress_wins} ({compress_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"ICL wins: {icl_wins} ({icl_wins/valid_comparisons*100:.1f}% of valid)")
    print(f"Ties: {ties} ({ties/valid_comparisons*100:.1f}% of valid)")
    print(f"Errors: {errors}")
    print(f"Total attempted: {total_comparisons}")
    print(f"Valid comparisons: {valid_comparisons}")
    
    if valid_comparisons > 0:
        if compress_wins > icl_wins:
            print(f"\n🎉 COMPRESS selection outperforms random by {compress_wins - icl_wins} wins!")
        elif icl_wins > compress_wins:
            print(f"\n⚠️  ICL selection outperforms COMPRESS by {icl_wins - compress_wins} wins")
        else:
            print(f"\n🤷 COMPRESS and ICL perform equally")
    else:
        print(f"\n❌ No valid comparisons completed")
    
    # Save results to file
    results_file = project_root / "results" / "compress_evaluation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump({
            "info": user,
            "results": {
                "compress_wins": compress_wins,
                "icl_wins": icl_wins,
                "ties": ties,
                "errors": errors,
                "total_attempted": total_comparisons,
                "valid_comparisons": valid_comparisons
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()