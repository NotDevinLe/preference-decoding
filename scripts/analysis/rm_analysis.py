#!/usr/bin/env python3
"""
Reward Matrix Analysis Script

Analyzes a .pt file to detect garbage data, anomalies, and validate structure.
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_reward_matrix(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of a reward matrix .pt file
    
    Args:
        file_path: Path to the .pt file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Analyzing reward matrix: {file_path}")
    
    # Load the tensor
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"✅ Successfully loaded tensor")
    except Exception as e:
        return {"error": f"Failed to load file: {e}"}
    
    # Basic structure analysis
    analysis = {
        "file_path": file_path,
        "tensor_shape": data.shape if hasattr(data, 'shape') else "Not a tensor",
        "tensor_dtype": data.dtype if hasattr(data, 'dtype') else "Unknown",
        "tensor_device": str(data.device) if hasattr(data, 'device') else "Unknown",
        "is_tensor": isinstance(data, torch.Tensor),
        "warnings": [],
        "errors": [],
        "statistics": {},
        "anomalies": []
    }
    
    if not isinstance(data, torch.Tensor):
        analysis["errors"].append("File does not contain a PyTorch tensor")
        return analysis
    
    # Shape validation
    if len(data.shape) != 2:
        analysis["errors"].append(f"Expected 2D tensor, got {len(data.shape)}D")
        return analysis
    
    N, D = data.shape
    analysis["N"] = N
    analysis["D"] = D
    analysis["total_elements"] = N * D
    
    print(f"�� Matrix dimensions: {N} x {D}")
    print(f"📊 Total elements: {N * D:,}")
    
    # Convert to numpy for analysis
    try:
        np_data = data.numpy()
    except Exception as e:
        analysis["errors"].append(f"Failed to convert to numpy: {e}")
        return analysis
    
    # Statistical analysis
    analysis["statistics"] = {
        "mean": float(np.mean(np_data)),
        "std": float(np.std(np_data)),
        "min": float(np.min(np_data)),
        "max": float(np.max(np_data)),
        "median": float(np.median(np_data)),
        "q25": float(np.percentile(np_data, 25)),
        "q75": float(np.percentile(np_data, 75)),
        "nan_count": int(np.isnan(np_data).sum()),
        "inf_count": int(np.isinf(np_data).sum()),
        "zero_count": int((np_data == 0).sum()),
        "negative_count": int((np_data < 0).sum()),
        "positive_count": int((np_data > 0).sum())
    }
    
    stats = analysis["statistics"]
    print(f"�� Statistics:")
    print(f"   Mean: {stats['mean']:.6f}")
    print(f"   Std:  {stats['std']:.6f}")
    print(f"   Min:  {stats['min']:.6f}")
    print(f"   Max:  {stats['max']:.6f}")
    print(f"   NaN:  {stats['nan_count']:,}")
    print(f"   Inf:  {stats['inf_count']:,}")
    
    # Garbage detection
    if stats['nan_count'] > 0:
        analysis["errors"].append(f"Found {stats['nan_count']} NaN values - this is garbage!")
    
    if stats['inf_count'] > 0:
        analysis["errors"].append(f"Found {stats['inf_count']} infinite values - this is garbage!")
    
    # Check for suspicious patterns
    if stats['std'] == 0:
        analysis["warnings"].append("All values are identical - might be garbage")
    
    if stats['min'] == stats['max']:
        analysis["warnings"].append("All values are the same - might be garbage")
    
    # Check for extreme values
    if abs(stats['mean']) > 1000:
        analysis["warnings"].append(f"Mean is very large: {stats['mean']:.2f}")
    
    if stats['std'] > 1000:
        analysis["warnings"].append(f"Standard deviation is very large: {stats['std']:.2f}")
    
    # Check for reasonable reward ranges (typically -10 to +10)
    if stats['min'] < -100 or stats['max'] > 100:
        analysis["warnings"].append(f"Values outside typical reward range [-100, 100]")
    
    # Check for too many zeros
    zero_ratio = stats['zero_count'] / stats['total_elements']
    if zero_ratio > 0.5:
        analysis["warnings"].append(f"High ratio of zeros: {zero_ratio:.2%}")
    
    # Check for constant rows/columns
    row_std = np.std(np_data, axis=1)
    col_std = np.std(np_data, axis=0)
    
    constant_rows = np.sum(row_std == 0)
    constant_cols = np.sum(col_std == 0)
    
    if constant_rows > 0:
        analysis["warnings"].append(f"Found {constant_rows} constant rows")
    
    if constant_cols > 0:
        analysis["warnings"].append(f"Found {constant_cols} constant columns")
    
    # Check for patterns that might indicate errors
    if stats['negative_count'] == stats['total_elements']:
        analysis["warnings"].append("All values are negative - might be an error")
    
    if stats['positive_count'] == stats['total_elements']:
        analysis["warnings"].append("All values are positive - might be an error")
    
    # Check for suspicious distributions
    if stats['std'] < 0.001 and stats['std'] > 0:
        analysis["warnings"].append("Very low variance - might be garbage")
    
    # Row-wise analysis
    row_means = np.mean(np_data, axis=1)
    row_stds = np.std(np_data, axis=1)
    
    analysis["row_analysis"] = {
        "mean_of_means": float(np.mean(row_means)),
        "std_of_means": float(np.std(row_means)),
        "mean_of_stds": float(np.mean(row_stds)),
        "std_of_stds": float(np.std(row_stds))
    }
    
    # Column-wise analysis
    col_means = np.mean(np_data, axis=0)
    col_stds = np.std(np_data, axis=0)
    
    analysis["col_analysis"] = {
        "mean_of_means": float(np.mean(col_means)),
        "std_of_means": float(np.std(col_means)),
        "mean_of_stds": float(np.mean(col_stds)),
        "std_of_stds": float(np.std(col_stds))
    }
    
    # Check for suspicious row/column patterns
    if analysis["row_analysis"]["std_of_means"] < 0.001:
        analysis["warnings"].append("Rows have very similar means - might be garbage")
    
    if analysis["col_analysis"]["std_of_means"] < 0.001:
        analysis["warnings"].append("Columns have very similar means - might be garbage")
    
    # Overall assessment
    if analysis["errors"]:
        analysis["assessment"] = "❌ GARBAGE - Contains errors"
    elif analysis["warnings"]:
        analysis["assessment"] = "⚠️  SUSPICIOUS - Has warnings"
    else:
        analysis["assessment"] = "✅ LOOKS GOOD - No obvious issues"
    
    return analysis

def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results"""
    print(f"\n{'='*60}")
    print(f"📋 ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"File: {analysis['file_path']}")
    print(f"Shape: {analysis['tensor_shape']}")
    print(f"Type: {analysis['tensor_dtype']}")
    print(f"Device: {analysis['tensor_device']}")
    
    print(f"\n�� STATISTICS:")
    stats = analysis['statistics']
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std:  {stats['std']:.6f}")
    print(f"  Min:  {stats['min']:.6f}")
    print(f"  Max:  {stats['max']:.6f}")
    print(f"  NaN:  {stats['nan_count']:,}")
    print(f"  Inf:  {stats['inf_count']:,}")
    print(f"  Zeros: {stats['zero_count']:,} ({stats['zero_count']/stats['total_elements']:.1%})")
    
    if analysis['errors']:
        print(f"\n❌ ERRORS:")
        for error in analysis['errors']:
            print(f"  - {error}")
    
    if analysis['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in analysis['warnings']:
            print(f"  - {warning}")
    
    print(f"\n🎯 ASSESSMENT: {analysis['assessment']}")
    
    if analysis['assessment'] == "✅ LOOKS GOOD - No obvious issues":
        print(f"✅ Your reward matrix appears to be valid!")
    elif analysis['assessment'] == "⚠️  SUSPICIOUS - Has warnings":
        print(f"⚠️  Your reward matrix has some suspicious patterns - review the warnings above")
    else:
        print(f"❌ Your reward matrix contains garbage data - check the errors above")

def save_analysis(analysis: Dict[str, Any], output_path: str):
    """Save analysis results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean up the analysis for JSON
    clean_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, dict):
            clean_analysis[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            clean_analysis[key] = convert_numpy(value)
    
    with open(output_path, 'w') as f:
        json.dump(clean_analysis, f, indent=2)
    
    print(f"�� Analysis saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze reward matrix .pt file for garbage data")
    parser.add_argument("--file_path", type=str, help="Path to the .pt file to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file for analysis results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"❌ File not found: {args.file_path}")
        return
    
    # Run analysis
    analysis = analyze_reward_matrix(args.file_path)
    
    # Print results
    print_analysis(analysis)
    
    # Save results if requested
    if args.output:
        save_analysis(analysis, args.output)
    
    # Exit with error code if garbage detected
    if analysis['assessment'] == "❌ GARBAGE - Contains errors":
        exit(1)
    elif analysis['assessment'] == "⚠️  SUSPICIOUS - Has warnings":
        exit(2)
    else:
        exit(0)

if __name__ == "__main__":
    main()