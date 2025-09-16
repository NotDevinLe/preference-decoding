#!/usr/bin/env python3
"""
Results Analysis Script
Analyzes CSV results from test_approximation_sweep.py and test_approximation_random.py
to generate comparison tables and winrate matrices.

This script generates:
1. 2D winrate table: Lambda values (rows) vs Users (columns) showing accuracy percentages
2. Sparse vs Random comparison table showing which method wins for each user/lambda combination
3. Summary statistics and performance comparisons

Usage examples:
    # Analyze sparse selection results only
    python analyze_results.py --sparse-csv approximation_sweep_multi_user1_5_20250915_123456.csv

    # Compare sparse vs random results
    python analyze_results.py --sparse-csv sparse_results.csv --random-csv random_results.csv

    # Generate tables with custom formatting
    python analyze_results.py --sparse-csv sparse_results.csv --random-csv random_results.csv --precision 3

    # Save analysis to file
    python analyze_results.py --sparse-csv sparse_results.csv --random-csv random_results.csv --output analysis_results.txt
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def load_csv_results(csv_path):
    """Load and validate CSV results"""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} results from {csv_path}")
    
    # Validate required columns
    required_cols = ['user', 'l1_lambda', 'accuracy', 'success']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter only successful results
    success_df = df[df['success'] == True].copy()
    failed_count = len(df) - len(success_df)
    if failed_count > 0:
        print(f"⚠️  Filtered out {failed_count} failed results")
    
    return success_df

def get_best_accuracy_per_user_lambda(df, method_name=""):
    """Get best accuracy for each user/lambda combination"""
    if 'sparsity_key' in df.columns:
        # For sparse results, group by user, lambda, and sparsity_key, then take best
        grouped = df.groupby(['user', 'l1_lambda', 'sparsity_key'])['accuracy'].max().reset_index()
        # Then take best across sparsity keys for each user/lambda
        best_df = grouped.groupby(['user', 'l1_lambda'])['accuracy'].max().reset_index()
    elif 'num_attributes' in df.columns and 'test_id' in df.columns:
        # For random results, group by user, lambda, num_attributes, then take best across test_ids
        grouped = df.groupby(['user', 'l1_lambda', 'num_attributes'])['accuracy'].max().reset_index()
        # Then take best across attribute counts for each user/lambda
        best_df = grouped.groupby(['user', 'l1_lambda'])['accuracy'].max().reset_index()
    else:
        # Simple case: just take best accuracy for each user/lambda
        best_df = df.groupby(['user', 'l1_lambda'])['accuracy'].max().reset_index()
    
    print(f"📈 {method_name} best results: {len(best_df)} user/lambda combinations")
    return best_df

def create_winrate_table(df, title="Winrate Table", precision=3):
    """Create 2D table with lambda (rows) vs users (columns) showing accuracy"""
    # Pivot the data
    pivot_df = df.pivot(index='l1_lambda', columns='user', values='accuracy')
    
    # Convert to percentages and format
    pivot_df = pivot_df * 100  # Convert to percentage
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print("Accuracy (%) by Lambda (rows) vs Users (columns)")
    print(f"{'='*60}")
    
    # Format and display the table
    formatted_df = pivot_df.round(precision)
    
    # Print with nice formatting
    print(formatted_df.to_string(float_format=f'%.{precision}f'))
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Average accuracy: {pivot_df.mean().mean():.{precision}f}%")
    print(f"Best accuracy: {pivot_df.max().max():.{precision}f}%")
    print(f"Worst accuracy: {pivot_df.min().min():.{precision}f}%")
    print(f"Standard deviation: {pivot_df.std().mean():.{precision}f}%")
    
    return pivot_df

def create_comparison_table(sparse_df, random_df, precision=3):
    """Create comparison table showing sparse vs random winrates"""
    # Merge the dataframes on user and l1_lambda
    merged = pd.merge(sparse_df, random_df, on=['user', 'l1_lambda'], suffixes=('_sparse', '_random'))
    
    # Calculate which method wins
    merged['sparse_wins'] = merged['accuracy_sparse'] > merged['accuracy_random']
    merged['accuracy_diff'] = (merged['accuracy_sparse'] - merged['accuracy_random']) * 100  # Convert to percentage points
    
    print(f"\n{'='*80}")
    print("SPARSE vs RANDOM COMPARISON")
    print(f"{'='*80}")
    print("Showing accuracy differences (Sparse - Random) in percentage points")
    print("Positive values = Sparse wins, Negative values = Random wins")
    print(f"{'='*80}")
    
    # Create pivot table for accuracy differences
    diff_pivot = merged.pivot(index='l1_lambda', columns='user', values='accuracy_diff')
    formatted_diff = diff_pivot.round(precision)
    print(formatted_diff.to_string(float_format=f'%+.{precision}f'))
    
    # Create winrate summary
    winrate_pivot = merged.pivot(index='l1_lambda', columns='user', values='sparse_wins')
    
    print(f"\n📊 Sparse Selection Wins (True/False):")
    print(winrate_pivot.to_string())
    
    # Overall statistics
    total_comparisons = len(merged)
    sparse_wins = merged['sparse_wins'].sum()
    random_wins = total_comparisons - sparse_wins
    
    print(f"\n🏆 Overall Performance:")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Sparse wins: {sparse_wins} ({sparse_wins/total_comparisons*100:.1f}%)")
    print(f"Random wins: {random_wins} ({random_wins/total_comparisons*100:.1f}%)")
    print(f"Average accuracy difference: {merged['accuracy_diff'].mean():.{precision}f} percentage points")
    print(f"Largest sparse advantage: +{merged['accuracy_diff'].max():.{precision}f} percentage points")
    print(f"Largest random advantage: {merged['accuracy_diff'].min():.{precision}f} percentage points")
    
    return merged, diff_pivot, winrate_pivot

def analyze_by_lambda(sparse_df, random_df=None, precision=3):
    """Analyze performance by lambda value"""
    print(f"\n{'='*60}")
    print("ANALYSIS BY LAMBDA VALUE")
    print(f"{'='*60}")
    
    # Sparse analysis
    sparse_by_lambda = sparse_df.groupby('l1_lambda')['accuracy'].agg(['mean', 'std', 'min', 'max']).round(precision+2)
    print(f"\n📈 Sparse Selection Performance by Lambda:")
    print(sparse_by_lambda.to_string())
    
    if random_df is not None:
        # Random analysis
        random_by_lambda = random_df.groupby('l1_lambda')['accuracy'].agg(['mean', 'std', 'min', 'max']).round(precision+2)
        print(f"\n🎲 Random Selection Performance by Lambda:")
        print(random_by_lambda.to_string())
        
        # Best lambda for each method
        sparse_best_lambda = sparse_by_lambda['mean'].idxmax()
        random_best_lambda = random_by_lambda['mean'].idxmax()
        
        print(f"\n🏆 Best Lambda Values:")
        print(f"Sparse selection: λ = {sparse_best_lambda} (avg accuracy: {sparse_by_lambda.loc[sparse_best_lambda, 'mean']:.{precision}f})")
        print(f"Random selection: λ = {random_best_lambda} (avg accuracy: {random_by_lambda.loc[random_best_lambda, 'mean']:.{precision}f})")

def analyze_by_user(sparse_df, random_df=None, precision=3):
    """Analyze performance by user"""
    print(f"\n{'='*60}")
    print("ANALYSIS BY USER")
    print(f"{'='*60}")
    
    # Sparse analysis
    sparse_by_user = sparse_df.groupby('user')['accuracy'].agg(['mean', 'std', 'min', 'max']).round(precision+2)
    print(f"\n📈 Sparse Selection Performance by User:")
    print(sparse_by_user.to_string())
    
    if random_df is not None:
        # Random analysis
        random_by_user = random_df.groupby('user')['accuracy'].agg(['mean', 'std', 'min', 'max']).round(precision+2)
        print(f"\n🎲 Random Selection Performance by User:")
        print(random_by_user.to_string())
        
        # Best users for each method
        sparse_best_user = sparse_by_user['mean'].idxmax()
        random_best_user = random_by_user['mean'].idxmax()
        
        print(f"\n🏆 Best Performing Users:")
        print(f"Sparse selection: {sparse_best_user} (avg accuracy: {sparse_by_user.loc[sparse_best_user, 'mean']:.{precision}f})")
        print(f"Random selection: {random_best_user} (avg accuracy: {random_by_user.loc[random_best_user, 'mean']:.{precision}f})")

def main():
    parser = argparse.ArgumentParser(description="Analyze approximation test results from CSV files")
    parser.add_argument("--sparse-csv", type=str, required=True, help="CSV file from sparse selection (test_approximation_sweep.py)")
    parser.add_argument("--random-csv", type=str, help="CSV file from random selection (test_approximation_random.py)")
    parser.add_argument("--precision", type=int, default=3, help="Decimal places for formatting (default: 3)")
    parser.add_argument("--output", type=str, help="Save analysis to text file")
    
    args = parser.parse_args()
    
    # Redirect output to file if requested
    if args.output:
        original_stdout = sys.stdout
        sys.stdout = open(args.output, 'w')
        print(f"Analysis Results - Generated on {pd.Timestamp.now()}")
        print(f"Sparse CSV: {args.sparse_csv}")
        if args.random_csv:
            print(f"Random CSV: {args.random_csv}")
        print("="*80)
    
    try:
        # Load sparse results
        print(f"🚀 ANALYZING APPROXIMATION RESULTS")
        print(f"Sparse selection file: {args.sparse_csv}")
        sparse_df = load_csv_results(args.sparse_csv)
        sparse_best = get_best_accuracy_per_user_lambda(sparse_df, "Sparse")
        
        # Create sparse winrate table
        sparse_pivot = create_winrate_table(sparse_best, "SPARSE SELECTION WINRATE TABLE", args.precision)
        
        # Analyze sparse results
        analyze_by_lambda(sparse_best, precision=args.precision)
        analyze_by_user(sparse_best, precision=args.precision)
        
        # Load random results if provided
        if args.random_csv:
            print(f"\nRandom selection file: {args.random_csv}")
            random_df = load_csv_results(args.random_csv)
            random_best = get_best_accuracy_per_user_lambda(random_df, "Random")
            
            # Create random winrate table
            random_pivot = create_winrate_table(random_best, "RANDOM SELECTION WINRATE TABLE", args.precision)
            
            # Create comparison tables
            merged_df, diff_pivot, winrate_pivot = create_comparison_table(sparse_best, random_best, args.precision)
            
            # Additional analyses
            analyze_by_lambda(sparse_best, random_best, args.precision)
            analyze_by_user(sparse_best, random_best, args.precision)
            
        print(f"\n✅ ANALYSIS COMPLETE")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"💾 Analysis saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())