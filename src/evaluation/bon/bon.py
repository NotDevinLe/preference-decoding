#!/usr/bin/env python3
"""
BON (Best-of-N) evaluation using persona scores from LLM judge.
Similar to drift_bon.py but uses persona evaluations instead of golden reward model.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_judge import PersonaJudge
from persona_rubric import extract_persona_from_prompt


def load_bon_data(data_path: str) -> List[Dict]:
    """Load BON dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def load_persona_evaluations(eval_path: str) -> Dict[str, Dict]:
    """
    Load persona evaluations and organize by prompt and response.
    Returns nested dict: {prompt: {response: evaluation}}
    """
    evaluations = defaultdict(dict)
    
    if not os.path.exists(eval_path):
        print(f"Warning: Evaluation file {eval_path} not found")
        return evaluations
    
    with open(eval_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry['prompt']
            response = entry['response']
            evaluations[prompt][response] = entry
    
    return evaluations


def evaluate_with_persona_scores(
    bon_data: List[Dict],
    evaluations: Dict[str, Dict],
    n_values: List[int],
    score_dimension: str = 'overall',
    aggregation: str = 'mean',
    output_path: str = None
) -> List[Dict]:
    """
    Evaluate BON performance using persona scores.
    
    Args:
        bon_data: BON dataset
        evaluations: Pre-computed persona evaluations
        n_values: List of n values for best-of-n
        score_dimension: Which score dimension to use for selection
        aggregation: How to aggregate scores ('mean', 'weighted', 'min')
        output_path: Path to save results
    
    Returns:
        List of result dictionaries
    """
    
    results = []
    
    for n in n_values:
        print(f"\n--- Evaluating with n={n} outputs ---")
        
        selected_scores = []
        random_scores = []
        oracle_scores = []  # Best possible selection
        
        missing_count = 0
        
        for item in bon_data:
            prompt = item['prompt']
            outputs = item['outputs'][:n]  # Use only first n outputs
            
            # Get scores for all outputs
            output_scores = []
            valid_outputs = []
            
            for output in outputs:
                if prompt in evaluations and output in evaluations[prompt]:
                    eval_entry = evaluations[prompt][output]
                    
                    # Get the appropriate score
                    if 'scores' in eval_entry:
                        if score_dimension in eval_entry['scores']:
                            score = eval_entry['scores'][score_dimension]
                        else:
                            # Fallback to overall if dimension not found
                            score = eval_entry['scores'].get('overall', 3.0)
                    else:
                        score = 3.0  # Default middle score
                    
                    output_scores.append(score)
                    valid_outputs.append(output)
                else:
                    missing_count += 1
            
            if not output_scores:
                continue
            
            # Get scores array
            scores_array = np.array(output_scores)
            
            # Best selection (oracle)
            best_idx = np.argmax(scores_array)
            oracle_scores.append(scores_array[best_idx])
            
            # For this simplified version, we'll use the first output as "selected"
            # In the full implementation, this would use your drift model's selection
            selected_scores.append(scores_array[0])
            
            # Random selection
            random_idx = np.random.randint(len(scores_array))
            random_scores.append(scores_array[random_idx])
        
        if missing_count > 0:
            print(f"  Warning: {missing_count} outputs missing persona evaluations")
        
        # Calculate statistics
        if selected_scores:
            result = {
                'n': n,
                'score_dimension': score_dimension,
                'num_prompts': len(selected_scores),
                'selected_mean': np.mean(selected_scores),
                'selected_std': np.std(selected_scores),
                'random_mean': np.mean(random_scores),
                'random_std': np.std(random_scores),
                'oracle_mean': np.mean(oracle_scores),
                'oracle_std': np.std(oracle_scores),
                'improvement_over_random': np.mean(selected_scores) - np.mean(random_scores),
                'oracle_gap': np.mean(oracle_scores) - np.mean(selected_scores)
            }
            
            results.append(result)
            
            print(f"  Results for n={n}:")
            print(f"    Selected: {result['selected_mean']:.3f} (±{result['selected_std']:.3f})")
            print(f"    Random:   {result['random_mean']:.3f} (±{result['random_std']:.3f})")
            print(f"    Oracle:   {result['oracle_mean']:.3f} (±{result['oracle_std']:.3f})")
            print(f"    Improvement: {result['improvement_over_random']:.3f}")
            print(f"    Gap to oracle: {result['oracle_gap']:.3f}")
    
    # Save results if path provided
    if output_path:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"\nResults saved to: {output_path}")
    
    return results


def evaluate_with_online_scoring(
    bon_data: List[Dict],
    judge: PersonaJudge,
    n_values: List[int],
    max_prompts: Optional[int] = None,
    output_path: str = None
) -> List[Dict]:
    """
    Evaluate BON with online persona scoring (no pre-computed evaluations).
    
    Args:
        bon_data: BON dataset
        judge: PersonaJudge instance
        n_values: List of n values for best-of-n
        max_prompts: Maximum number of prompts to evaluate
        output_path: Path to save results
    
    Returns:
        List of result dictionaries
    """
    
    if max_prompts:
        bon_data = bon_data[:max_prompts]
    
    results = []
    
    for n in n_values:
        print(f"\n--- Evaluating with n={n} outputs (online scoring) ---")
        
        all_scores = []
        
        for i, item in enumerate(bon_data):
            prompt = item['prompt']
            outputs = item['outputs'][:n]
            
            # Extract persona from prompt
            persona = extract_persona_from_prompt(prompt)
            
            print(f"  [{i+1}/{len(bon_data)}] Evaluating {len(outputs)} outputs for prompt...")
            
            # Score all outputs
            scores = []
            for output in outputs:
                score = judge.score_response(persona, prompt, output)
                if score:
                    scores.append(score.get_overall())
                else:
                    scores.append(3.0)  # Default middle score
            
            all_scores.append(scores)
        
        # Calculate statistics
        selected_scores = [scores[0] for scores in all_scores]  # First output
        random_scores = [np.random.choice(scores) for scores in all_scores]
        oracle_scores = [max(scores) for scores in all_scores]
        
        result = {
            'n': n,
            'num_prompts': len(all_scores),
            'selected_mean': np.mean(selected_scores),
            'selected_std': np.std(selected_scores),
            'random_mean': np.mean(random_scores),
            'random_std': np.std(random_scores),
            'oracle_mean': np.mean(oracle_scores),
            'oracle_std': np.std(oracle_scores),
            'improvement_over_random': np.mean(selected_scores) - np.mean(random_scores),
            'oracle_gap': np.mean(oracle_scores) - np.mean(selected_scores)
        }
        
        results.append(result)
        
        print(f"\n  Summary for n={n}:")
        print(f"    Selected: {result['selected_mean']:.3f} (±{result['selected_std']:.3f})")
        print(f"    Random:   {result['random_mean']:.3f} (±{result['random_std']:.3f})")
        print(f"    Oracle:   {result['oracle_mean']:.3f} (±{result['oracle_std']:.3f})")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"\nResults saved to: {output_path}")
    
    # Print judge statistics
    print(f"\n=== JUDGE STATISTICS ===")
    stats = judge.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return results


def print_summary_table(results: List[Dict]):
    """Print a summary table of results."""
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'n':<5} {'Selected':<15} {'Random':<15} {'Oracle':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['n']:<5} "
              f"{r['selected_mean']:.3f} (±{r['selected_std']:.2f})  "
              f"{r['random_mean']:.3f} (±{r['random_std']:.2f})  "
              f"{r['oracle_mean']:.3f} (±{r['oracle_std']:.2f})  "
              f"{r['improvement_over_random']:+.3f}")
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BON evaluation with persona scores")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/bon.json",
        help="Path to BON dataset"
    )
    
    parser.add_argument(
        "--eval_path",
        type=str,
        default="../../results/persona_evaluations_bon.jsonl",
        help="Path to pre-computed persona evaluations"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="../../results/persona_bon_results.jsonl",
        help="Path to save results"
    )
    
    parser.add_argument(
        "--n_values",
        type=str,
        default="5,10,20,50,100",
        help="Comma-separated list of n values for best-of-n"
    )
    
    parser.add_argument(
        "--score_dimension",
        type=str,
        default="overall",
        choices=['overall', 'speaking_style', 'personality', 'knowledge', 'behavioral', 'emotional'],
        help="Which score dimension to use for selection"
    )
    
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use online scoring instead of pre-computed evaluations"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Parse n values
    n_values = [int(x) for x in args.n_values.split(',')]
    
    # Load BON data
    print(f"Loading BON data from: {args.data_path}")
    bon_data = load_bon_data(args.data_path)
    print(f"Loaded {len(bon_data)} prompts")
    
    if args.online:
        # Online scoring mode
        print("\nUsing online persona scoring...")
        judge = PersonaJudge()
        results = evaluate_with_online_scoring(
            bon_data,
            judge,
            n_values,
            max_prompts=args.max_prompts,
            output_path=args.output_path
        )
    else:
        # Pre-computed evaluations mode
        print(f"\nLoading persona evaluations from: {args.eval_path}")
        evaluations = load_persona_evaluations(args.eval_path)
        
        # Count evaluations
        total_evals = sum(len(responses) for responses in evaluations.values())
        print(f"Loaded evaluations for {len(evaluations)} prompts ({total_evals} total)")
        
        results = evaluate_with_persona_scores(
            bon_data,
            evaluations,
            n_values,
            score_dimension=args.score_dimension,
            output_path=args.output_path
        )
    
    # Print summary
    print_summary_table(results)