#!/usr/bin/env python3
"""
Analysis tools for persona evaluations.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict


def load_evaluations(filepath: str) -> List[Dict]:
    """Load evaluations from JSONL file."""
    evaluations = []
    with open(filepath, 'r') as f:
        for line in f:
            evaluations.append(json.loads(line))
    return evaluations


def evaluations_to_dataframe(evaluations: List[Dict]) -> pd.DataFrame:
    """Convert evaluations to pandas DataFrame for easier analysis."""
    
    records = []
    for eval_entry in evaluations:
        record = {
            'prompt_idx': eval_entry.get('prompt_idx', -1),
            'output_idx': eval_entry.get('output_idx', -1),
            'prompt': eval_entry['prompt'][:100],  # Truncate for display
            'persona': eval_entry.get('persona', '')[:100],
            'response_length': len(eval_entry.get('response', '')),
        }
        
        # Add scores
        if 'scores' in eval_entry:
            for key, value in eval_entry['scores'].items():
                record[f'score_{key}'] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


def analyze_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlations between different scoring dimensions."""
    
    score_columns = [col for col in df.columns if col.startswith('score_')]
    if len(score_columns) < 2:
        print("Not enough score columns for correlation analysis")
        return pd.DataFrame()
    
    correlation_matrix = df[score_columns].corr()
    
    # Pretty print correlation matrix
    print("\n=== DIMENSION CORRELATIONS ===")
    print(correlation_matrix.to_string())
    
    return correlation_matrix


def analyze_score_distributions(df: pd.DataFrame, save_plots: bool = False, output_dir: str = "../results/plots"):
    """Analyze and visualize score distributions."""
    
    score_columns = [col for col in df.columns if col.startswith('score_') and col != 'score_overall']
    
    if not score_columns:
        print("No score columns found")
        return
    
    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subplots for each dimension
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(score_columns):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        dimension_name = col.replace('score_', '').replace('_', ' ').title()
        
        # Plot histogram
        ax.hist(df[col], bins=5, range=(1, 6), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{dimension_name} Distribution')
        ax.set_xticks([1, 2, 3, 4, 5])
        
        # Add mean line
        mean_score = df[col].mean()
        ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.2f}')
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(score_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Score Distributions Across Dimensions')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/score_distributions.png", dpi=150)
        print(f"Saved plot to {output_dir}/score_distributions.png")
    else:
        plt.show()
    
    # Print statistics
    print("\n=== SCORE STATISTICS ===")
    for col in score_columns:
        dimension_name = col.replace('score_', '').replace('_', ' ').title()
        print(f"\n{dimension_name}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Std:  {df[col].std():.3f}")
        print(f"  Min:  {df[col].min()}")
        print(f"  Max:  {df[col].max()}")
        print(f"  Median: {df[col].median()}")


def find_best_worst_examples(evaluations: List[Dict], n: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Find best and worst scoring examples."""
    
    # Sort by overall score
    sorted_evals = sorted(evaluations, key=lambda x: x.get('scores', {}).get('overall', 0))
    
    worst = sorted_evals[:n]
    best = sorted_evals[-n:]
    
    return best, worst


def analyze_by_persona_type(evaluations: List[Dict]) -> Dict[str, Dict]:
    """Analyze performance grouped by persona type."""
    
    # Group by persona
    persona_groups = defaultdict(list)
    for eval_entry in evaluations:
        persona = eval_entry.get('persona', 'Unknown')
        # Simplify persona to key words for grouping
        persona_key = persona[:50]  # Use first 50 chars as key
        persona_groups[persona_key].append(eval_entry)
    
    # Analyze each group
    results = {}
    for persona_key, evals in persona_groups.items():
        scores = {
            'speaking_style': [],
            'personality': [],
            'knowledge': [],
            'behavioral': [],
            'emotional': [],
            'overall': []
        }
        
        for eval_entry in evals:
            if 'scores' in eval_entry:
                for key in scores:
                    if key in eval_entry['scores']:
                        scores[key].append(eval_entry['scores'][key])
        
        # Calculate statistics
        stats_dict = {}
        for key, values in scores.items():
            if values:
                stats_dict[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        results[persona_key] = {
            'stats': stats_dict,
            'count': len(evals)
        }
    
    return results


def compare_dimension_importance(df: pd.DataFrame) -> Dict[str, float]:
    """
    Determine which dimensions are most predictive of overall score.
    Uses regression analysis to find feature importance.
    """
    
    if 'score_overall' not in df.columns:
        print("No overall score column found")
        return {}
    
    dimension_columns = [col for col in df.columns if col.startswith('score_') 
                        and col not in ['score_overall']]
    
    if not dimension_columns:
        print("No dimension score columns found")
        return {}
    
    # Prepare data
    X = df[dimension_columns].values
    y = df['score_overall'].values
    
    # Remove any rows with NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        print("Not enough data for regression analysis")
        return {}
    
    # Use simple linear regression to get coefficients
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features for fair comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Get feature importance (coefficients)
    importance = {}
    for col, coef in zip(dimension_columns, model.coef_):
        dimension_name = col.replace('score_', '')
        importance[dimension_name] = abs(coef)  # Use absolute value
    
    # Normalize to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v/total for k, v in importance.items()}
    
    return importance


def generate_summary_report(evaluations: List[Dict], output_path: Optional[str] = None):
    """Generate a comprehensive summary report."""
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("PERSONA EVALUATION SUMMARY REPORT")
    report_lines.append("=" * 60)
    
    # Basic statistics
    report_lines.append(f"\nTotal Evaluations: {len(evaluations)}")
    
    # Convert to DataFrame
    df = evaluations_to_dataframe(evaluations)
    
    # Overall statistics
    if 'score_overall' in df.columns:
        report_lines.append(f"\nOverall Score Statistics:")
        report_lines.append(f"  Mean: {df['score_overall'].mean():.3f}")
        report_lines.append(f"  Std:  {df['score_overall'].std():.3f}")
        report_lines.append(f"  Min:  {df['score_overall'].min()}")
        report_lines.append(f"  Max:  {df['score_overall'].max()}")
    
    # Dimension statistics
    report_lines.append("\nDimension Statistics:")
    for col in df.columns:
        if col.startswith('score_') and col != 'score_overall':
            dimension_name = col.replace('score_', '').replace('_', ' ').title()
            report_lines.append(f"\n  {dimension_name}:")
            report_lines.append(f"    Mean: {df[col].mean():.3f}")
            report_lines.append(f"    Std:  {df[col].std():.3f}")
    
    # Dimension importance
    importance = compare_dimension_importance(df)
    if importance:
        report_lines.append("\nDimension Importance (for overall score):")
        for dim, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {dim}: {imp:.2%}")
    
    # Best and worst examples
    best, worst = find_best_worst_examples(evaluations, n=3)
    
    report_lines.append("\nTop 3 Best Scoring Responses:")
    for i, eval_entry in enumerate(best, 1):
        score = eval_entry.get('scores', {}).get('overall', 0)
        response = eval_entry.get('response', '')[:200]
        report_lines.append(f"\n  {i}. Score: {score:.2f}")
        report_lines.append(f"     Response: {response}...")
    
    report_lines.append("\nTop 3 Worst Scoring Responses:")
    for i, eval_entry in enumerate(worst, 1):
        score = eval_entry.get('scores', {}).get('overall', 0)
        response = eval_entry.get('response', '')[:200]
        report_lines.append(f"\n  {i}. Score: {score:.2f}")
        report_lines.append(f"     Response: {response}...")
    
    # Correlation analysis
    corr_matrix = analyze_dimension_correlations(df)
    if not corr_matrix.empty:
        report_lines.append("\nKey Correlations:")
        # Find strongest correlations (excluding diagonal)
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append((
                    corr_matrix.columns[i].replace('score_', ''),
                    corr_matrix.columns[j].replace('score_', ''),
                    corr_matrix.iloc[i, j]
                ))
        
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
        for dim1, dim2, corr in corr_values[:3]:
            report_lines.append(f"  {dim1} <-> {dim2}: {corr:.3f}")
    
    report_lines.append("\n" + "=" * 60)
    
    # Print report
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_path}")
    
    return report_text


def plot_dimension_comparison(evaluations: List[Dict], dimension1: str, dimension2: str, 
                             save_path: Optional[str] = None):
    """Create scatter plot comparing two dimensions."""
    
    scores1 = []
    scores2 = []
    
    for eval_entry in evaluations:
        if 'scores' in eval_entry:
            if dimension1 in eval_entry['scores'] and dimension2 in eval_entry['scores']:
                scores1.append(eval_entry['scores'][dimension1])
                scores2.append(eval_entry['scores'][dimension2])
    
    if not scores1:
        print(f"No data found for dimensions {dimension1} and {dimension2}")
        return
    
    plt.figure(figsize=(8, 6))
    plt.scatter(scores1, scores2, alpha=0.5)
    plt.xlabel(dimension1.replace('_', ' ').title())
    plt.ylabel(dimension2.replace('_', ' ').title())
    plt.title(f'{dimension1} vs {dimension2}')
    
    # Add correlation
    correlation = np.corrcoef(scores1, scores2)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze persona evaluations")
    parser.add_argument(
        "--input",
        type=str,
        default="../results/persona_evaluations_bon.jsonl",
        help="Path to evaluations JSONL file"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save summary report"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="../results/plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    # Load evaluations
    print(f"Loading evaluations from: {args.input}")
    evaluations = load_evaluations(args.input)
    print(f"Loaded {len(evaluations)} evaluations")
    
    # Generate summary report
    generate_summary_report(evaluations, args.report)
    
    # Create DataFrame for additional analysis
    df = evaluations_to_dataframe(evaluations)
    
    # Generate plots if requested
    if args.plots:
        print("\nGenerating plots...")
        analyze_score_distributions(df, save_plots=True, output_dir=args.plot_dir)
        
        # Example dimension comparison
        plot_dimension_comparison(
            evaluations, 
            'speaking_style', 
            'personality',
            save_path=f"{args.plot_dir}/speaking_vs_personality.png"
        )
    
    # Analyze by persona type
    print("\n=== ANALYSIS BY PERSONA TYPE ===")
    persona_results = analyze_by_persona_type(evaluations)
    
    # Show top 5 personas by count
    sorted_personas = sorted(persona_results.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)[:5]
    
    for persona, data in sorted_personas:
        print(f"\nPersona: {persona}...")
        print(f"  Count: {data['count']}")
        if 'stats' in data and 'overall' in data['stats']:
            overall_stats = data['stats']['overall']
            print(f"  Overall Score Mean: {overall_stats['mean']:.3f} (±{overall_stats['std']:.3f})")