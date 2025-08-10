#!/usr/bin/env python3
"""
Compare different BON evaluation methods (drift, persona, random).

Usage:
    python scripts/evaluate/run_bon_comparison.py \
        --data_path data/bon.json \
        --persona_eval_path results/evaluations/persona_scores.jsonl \
        --n_values 5,10,20,50,100
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.bon.persona_bon import (
    load_bon_data,
    load_persona_evaluations,
    evaluate_with_persona_scores
)


def plot_comparison(results_dict, output_path):
    """Plot comparison of different methods."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Performance by N
    for method, results in results_dict.items():
        n_values = [r['n'] for r in results]
        scores = [r.get('selected_mean', r.get('avg_score', 0)) for r in results]
        ax1.plot(n_values, scores, marker='o', label=method)
    
    ax1.set_xlabel('N (number of outputs)')
    ax1.set_ylabel('Average Score')
    ax1.set_title('BON Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement over random
    for method, results in results_dict.items():
        if 'improvement_over_random' in results[0]:
            n_values = [r['n'] for r in results]
            improvements = [r['improvement_over_random'] for r in results]
            ax2.plot(n_values, improvements, marker='s', label=method)
    
    ax2.set_xlabel('N (number of outputs)')
    ax2.set_ylabel('Improvement over Random')
    ax2.set_title('Improvement over Random Selection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare different BON evaluation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/bon.json",
        help="Path to BON dataset"
    )
    
    parser.add_argument(
        "--persona_eval_path",
        type=str,
        default="results/evaluations/persona_scores.jsonl",
        help="Path to persona evaluations"
    )
    
    parser.add_argument(
        "--golden_cache_path",
        type=str,
        default="results/gold_scores_bon.jsonl",
        help="Path to golden score cache"
    )
    
    parser.add_argument(
        "--n_values",
        type=str,
        default="5,10,20,50,100",
        help="Comma-separated list of n values"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparisons",
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    
    args = parser.parse_args()
    
    # Parse n values
    n_values = [int(x) for x in args.n_values.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading BON data...")
    bon_data = load_bon_data(args.data_path)
    print(f"Loaded {len(bon_data)} prompts")
    
    results_dict = {}
    
    # Evaluate with persona scores
    if Path(args.persona_eval_path).exists():
        print("\n" + "=" * 70)
        print("PERSONA-BASED EVALUATION")
        print("=" * 70)
        
        evaluations = load_persona_evaluations(args.persona_eval_path)
        total_evals = sum(len(responses) for responses in evaluations.values())
        print(f"Loaded {total_evals} persona evaluations")
        
        persona_results = evaluate_with_persona_scores(
            bon_data,
            evaluations,
            n_values,
            score_dimension='overall',
            output_path=str(output_dir / "persona_bon_results.jsonl")
        )
        
        results_dict['Persona'] = persona_results
    
    # Load golden scores if available
    if Path(args.golden_cache_path).exists():
        print("\n" + "=" * 70)
        print("GOLDEN REWARD MODEL RESULTS")
        print("=" * 70)
        
        # This would load and process golden RM results
        # For now, just indicate it's available
        print(f"Golden cache found at: {args.golden_cache_path}")
    
    # Generate comparison report
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Print comparison table
    if results_dict:
        print(f"\n{'Method':<15} {'N=5':<10} {'N=10':<10} {'N=20':<10} {'N=50':<10} {'N=100':<10}")
        print("-" * 65)
        
        for method, results in results_dict.items():
            scores_by_n = {r['n']: r.get('selected_mean', r.get('avg_score', 0)) 
                          for r in results}
            row = f"{method:<15}"
            for n in [5, 10, 20, 50, 100]:
                if n in scores_by_n:
                    row += f"{scores_by_n[n]:<10.3f}"
                else:
                    row += f"{'N/A':<10}"
            print(row)
    
    # Generate plots if requested
    if args.plot and results_dict:
        plot_path = output_dir / "bon_comparison.png"
        plot_comparison(results_dict, plot_path)
    
    # Save combined results
    combined_path = output_dir / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()