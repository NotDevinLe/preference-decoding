#!/usr/bin/env python3
"""
Comparison script for multi-method evaluation.
Loads responses from multiple generation files and compares them using LLM judge.

Usage:
    python scripts/evaluate/run_comparison.py \
        --generated_files results/generations/bon-drift.json results/generations/bon-mle.json \
        --judge_model meta-llama/Llama-3.3-70B-Instruct \
        --judge_base_url http://localhost:8000/v1 \
        --persona "A helpful AI assistant" \
        --output_path results/comparisons/comparison_results.json
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.judges.llm_judge import PersonaJudge
from src.evaluation.judges.persona_rubric import (
    create_multi_comparison_prompt,
    parse_multi_comparison_response,
    PersonaScore
)


def load_generation_files(file_paths: List[str]) -> Dict[str, Dict]:
    """
    Load generation results from multiple files.
    
    Args:
        file_paths: List of paths to generation result JSON files
        
    Returns:
        Dictionary mapping method names to generation data
    """
    
    generation_data = {}
    
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        method = data['method']
        generation_data[method] = data
        
        print(f"  Method: {method}")
        print(f"  Prompts: {data['num_prompts']}")
        print(f"  Generated: {data['timestamp']}")
    
    return generation_data


def validate_generation_data(generation_data: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    """
    Validate that all generation files have consistent prompts.
    
    Args:
        generation_data: Dictionary of generation data per method
        
    Returns:
        Tuple of (common_prompts, method_names)
        
    Raises:
        ValueError: If prompts don't match across methods
    """
    
    method_names = list(generation_data.keys())
    if not method_names:
        raise ValueError("No generation data provided")
    
    # Get prompts from first method
    first_method = method_names[0]
    common_prompts = generation_data[first_method]['prompts']
    
    # Validate all methods have same prompts
    for method in method_names[1:]:
        method_prompts = generation_data[method]['prompts']
        
        if len(method_prompts) != len(common_prompts):
            raise ValueError(f"Method {method} has {len(method_prompts)} prompts, expected {len(common_prompts)}")
        
        # Check if prompts are in same order
        for i, (prompt1, prompt2) in enumerate(zip(common_prompts, method_prompts)):
            if prompt1 != prompt2:
                raise ValueError(f"Prompt mismatch at index {i} between {first_method} and {method}")
    
    print(f"✅ Validated {len(common_prompts)} common prompts across {len(method_names)} methods")
    return common_prompts, method_names


def create_comparison_batches(
    prompts: List[str],
    generation_data: Dict[str, Dict],
    method_names: List[str],
    batch_size: int = 10
) -> List[List[Dict]]:
    """
    Create batches of comparisons to process.
    
    Args:
        prompts: List of prompts
        generation_data: Generation data per method  
        method_names: List of method names
        batch_size: Number of prompts per batch
        
    Returns:
        List of batches, each containing comparison data
    """
    
    batches = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_data = []
        
        for j, prompt in enumerate(batch_prompts):
            prompt_idx = i + j
            
            # Get responses for this prompt from all methods
            responses = {}
            for method in method_names:
                responses[method] = generation_data[method]['responses'][prompt_idx]
            
            batch_data.append({
                'prompt': prompt,
                'responses': responses,
                'prompt_index': prompt_idx
            })
        
        batches.append(batch_data)
    
    return batches


def process_comparison_batch(
    judge: PersonaJudge,
    persona: str,
    batch_data: List[Dict],
    method_names: List[str]
) -> List[Dict]:
    """
    Process a batch of comparisons.
    
    Args:
        judge: PersonaJudge instance
        persona: Persona description for evaluation
        batch_data: List of comparison data for this batch
        method_names: List of method names
        
    Returns:
        List of comparison results
    """
    
    batch_results = []
    
    for item in batch_data:
        prompt = item['prompt']
        responses = item['responses']
        prompt_index = item['prompt_index']
        
        # Create multi-comparison prompt
        comparison_prompt = create_multi_comparison_prompt(persona, prompt, responses)
        
        # Call judge
        try:
            llm_response = judge._call_llm(comparison_prompt)
            
            if llm_response:
                # Parse response
                scores_list, ranking_order, ranking_reason = parse_multi_comparison_response(
                    llm_response, method_names
                )
                
                # Create result
                result = {
                    'prompt_index': prompt_index,
                    'prompt': prompt,
                    'method_names': method_names,
                    'responses': responses,
                    'scores': [score.to_dict() for score in scores_list],
                    'overall_scores': [score.get_overall() for score in scores_list],
                    'ranking_order': ranking_order,
                    'ranking_reason': ranking_reason,
                    'llm_response': llm_response,
                    'success': True
                }
            else:
                # Failed comparison
                result = {
                    'prompt_index': prompt_index,
                    'prompt': prompt,
                    'method_names': method_names,
                    'responses': responses,
                    'success': False,
                    'error': 'LLM call failed'
                }
        
        except Exception as e:
            # Error in processing
            result = {
                'prompt_index': prompt_index,
                'prompt': prompt,
                'method_names': method_names,
                'responses': responses,
                'success': False,
                'error': str(e)
            }
        
        batch_results.append(result)
    
    return batch_results


def aggregate_results(all_results: List[Dict], method_names: List[str]) -> Dict[str, Any]:
    """
    Aggregate comparison results into summary statistics.
    
    Args:
        all_results: List of all comparison results
        method_names: List of method names
        
    Returns:
        Dictionary with aggregated statistics
    """
    
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if not successful_results:
        return {
            'total_comparisons': len(all_results),
            'successful_comparisons': 0,
            'error': 'No successful comparisons'
        }
    
    # Calculate average scores per method
    method_scores = defaultdict(list)
    method_rankings = defaultdict(list)
    
    for result in successful_results:
        overall_scores = result['overall_scores']
        ranking_order = result['ranking_order']
        
        # Collect scores
        for i, method in enumerate(method_names):
            if i < len(overall_scores):
                method_scores[method].append(overall_scores[i])
        
        # Collect rankings (convert to rank positions: 1st place = 1, 2nd = 2, etc.)
        for rank_pos, response_num in enumerate(ranking_order, 1):
            method_idx = response_num - 1  # Convert to 0-indexed
            if 0 <= method_idx < len(method_names):
                method = method_names[method_idx]
                method_rankings[method].append(rank_pos)
    
    # Calculate statistics
    import numpy as np
    
    summary = {
        'total_comparisons': len(all_results),
        'successful_comparisons': len(successful_results),
        'method_statistics': {}
    }
    
    for method in method_names:
        scores = method_scores[method]
        rankings = method_rankings[method]
        
        if scores:
            summary['method_statistics'][method] = {
                'average_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'median_score': float(np.median(scores)),
                'average_rank': float(np.mean(rankings)) if rankings else None,
                'rank_std': float(np.std(rankings)) if rankings else None,
                'num_first_place': sum(1 for r in rankings if r == 1),
                'num_evaluations': len(scores)
            }
        else:
            summary['method_statistics'][method] = {
                'error': 'No successful evaluations'
            }
    
    # Overall ranking
    avg_ranks = {
        method: stats.get('average_rank', float('inf'))
        for method, stats in summary['method_statistics'].items()
        if 'average_rank' in stats
    }
    
    summary['overall_ranking'] = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    return summary


def save_comparison_results(
    all_results: List[Dict],
    summary: Dict[str, Any],
    method_names: List[str],
    args,
    output_path: str
):
    """Save comparison results to JSON file."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'persona': args.persona,
        'judge_model': args.judge_model,
        'method_names': method_names,
        'generation_files': args.generated_files,
        'parameters': {
            'max_prompts': getattr(args, 'max_prompts', None),
            'batch_size': getattr(args, 'batch_size', 10)
        },
        'summary': summary,
        'detailed_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary(summary: Dict[str, Any]):
    """Print a nice summary of the results."""
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS SUMMARY")
    print("="*80)
    
    print(f"Total Comparisons: {summary['total_comparisons']}")
    print(f"Successful: {summary['successful_comparisons']}")
    
    if summary.get('method_statistics'):
        print(f"\nMethod Performance (Average Score ± Std):")
        print("-" * 50)
        
        for method, stats in summary['method_statistics'].items():
            if 'average_score' in stats:
                score = stats['average_score']
                std = stats['std_score']
                rank = stats.get('average_rank', 'N/A')
                first_place = stats.get('num_first_place', 0)
                print(f"{method:<20} {score:.3f}±{std:.3f} (avg rank: {rank:.1f if rank != 'N/A' else rank}, 1st place: {first_place})")
            else:
                print(f"{method:<20} ERROR: {stats.get('error', 'Unknown')}")
        
        print(f"\nOverall Ranking (by average rank):")
        print("-" * 30)
        for i, (method, avg_rank) in enumerate(summary.get('overall_ranking', []), 1):
            print(f"{i}. {method} (avg rank: {avg_rank:.2f})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare responses from multiple generation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "--generated_files",
        nargs="+",
        required=True,
        help="Paths to generation result JSON files"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save comparison results"
    )
    
    # Judge configuration
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="LLM model to use as judge"
    )
    
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLLM endpoint URL for judge"
    )
    
    parser.add_argument(
        "--judge_cache_dir",
        type=str,
        default="cache/comparison_judge",
        help="Directory to cache judge responses"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Persona description for evaluation"
    )
    
    # Processing parameters
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to compare"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of prompts to process in each batch"
    )
    
    args = parser.parse_args()
    
    # Load generation files
    print("Loading generation files...")
    generation_data = load_generation_files(args.generated_files)
    
    # Validate data consistency
    common_prompts, method_names = validate_generation_data(generation_data)
    
    if args.max_prompts:
        common_prompts = common_prompts[:args.max_prompts]
        print(f"Limited to first {args.max_prompts} prompts")
    
    # Initialize judge
    print(f"\nInitializing judge ({args.judge_model})...")
    judge = PersonaJudge(
        base_url=args.judge_base_url,
        model=args.judge_model,
        cache_dir=args.judge_cache_dir,
        temperature=0.1,
        max_tokens=2048  # Longer for multi-response comparisons
    )
    
    print(f"Persona: {args.persona}")
    print(f"Methods to compare: {method_names}")
    
    # Create comparison batches
    print(f"\nCreating comparison batches...")
    batches = create_comparison_batches(common_prompts, generation_data, method_names, args.batch_size)
    print(f"Created {len(batches)} batches of up to {args.batch_size} prompts each")
    
    # Process batches
    print(f"\nProcessing comparisons...")
    all_results = []
    
    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{len(batches)} ({len(batch)} prompts)...")
        
        batch_results = process_comparison_batch(judge, args.persona, batch, method_names)
        all_results.extend(batch_results)
        
        # Show progress
        successful = sum(1 for r in batch_results if r.get('success', False))
        print(f"  Batch {i}: {successful}/{len(batch)} successful")
    
    # Aggregate results
    print(f"\nAggregating results...")
    summary = aggregate_results(all_results, method_names)
    
    # Save results
    save_comparison_results(all_results, summary, method_names, args, args.output_path)
    
    # Print summary
    print_summary(summary)
    
    # Print judge statistics
    if hasattr(judge, 'get_statistics'):
        stats = judge.get_statistics()
        print(f"\nJudge Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\n✅ Comparison complete!")
    print(f"Methods compared: {len(method_names)}")
    print(f"Prompts processed: {len(common_prompts)}")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()