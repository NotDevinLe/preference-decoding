#!/usr/bin/env python3
"""
Convert replay buffer from coordinator to format expected by gumbel.py analysis script.

Usage:
    python scripts/analysis/convert_buffer_to_gumbel.py --input replay_buffer.pkl --output reward_matrix_flexible.npz
    python scripts/analysis/convert_buffer_to_gumbel.py --input replay_buffer.pkl --output reward_matrix_flexible.npz --verbose
"""

import argparse
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_replay_buffer(buffer_path: str) -> Dict[str, Any]:
    """Load replay buffer from pickle file."""
    try:
        with open(buffer_path, 'rb') as f:
            buffer_data = pickle.load(f)
        return buffer_data
    except Exception as e:
        raise RuntimeError(f"Failed to load buffer from '{buffer_path}': {e}")


def extract_reward_matrix(buffer_data: Dict[str, Any]) -> np.ndarray:
    """Extract reward vectors from buffer and convert to 2D numpy array."""
    buffer_samples = buffer_data.get("buffer", [])
    
    if not buffer_samples:
        raise ValueError("Buffer is empty - no samples to convert")
    
    # Extract reward vectors
    reward_vectors = []
    for i, sample in enumerate(buffer_samples):
        if "reward_vector" not in sample:
            logging.warning(f"Sample {i} missing 'reward_vector' key, skipping")
            continue
            
        reward_vec = sample["reward_vector"]
        if not isinstance(reward_vec, (list, np.ndarray)):
            logging.warning(f"Sample {i} has invalid reward_vector type {type(reward_vec)}, skipping")
            continue
            
        reward_vectors.append(reward_vec)
    
    if not reward_vectors:
        raise ValueError("No valid reward vectors found in buffer")
    
    # Convert to numpy array
    try:
        reward_matrix = np.array(reward_vectors, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to convert reward vectors to numpy array: {e}")
    
    return reward_matrix


def save_for_gumbel_analysis(reward_matrix: np.ndarray, output_path: str) -> None:
    """Save reward matrix in format expected by gumbel.py."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as .npz file with expected key name
        np.savez(output_path, Y_chosen=reward_matrix)
        
    except Exception as e:
        raise RuntimeError(f"Failed to save to '{output_path}': {e}")


def analyze_buffer_data(buffer_data: Dict[str, Any], reward_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze buffer data and reward matrix statistics."""
    buffer_samples = buffer_data.get("buffer", [])
    
    # Basic buffer stats
    stats = {
        "buffer_size": len(buffer_samples),
        "buffer_maxlen": buffer_data.get("maxlen", "unknown"),
        "total_samples_collected": buffer_data.get("total_samples", "unknown"),
        "batch_count": buffer_data.get("batch_count", "unknown"),
    }
    
    # Reward matrix stats
    if reward_matrix.size > 0:
        stats.update({
            "reward_matrix_shape": reward_matrix.shape,
            "n_samples": reward_matrix.shape[0],
            "n_attributes": reward_matrix.shape[1],
            "reward_mean": float(reward_matrix.mean()),
            "reward_std": float(reward_matrix.std()),
            "reward_min": float(reward_matrix.min()),
            "reward_max": float(reward_matrix.max()),
        })
        
        # Check for NaN/inf values
        n_nan = np.isnan(reward_matrix).sum()
        n_inf = np.isinf(reward_matrix).sum()
        stats.update({
            "n_nan_values": int(n_nan),
            "n_inf_values": int(n_inf),
            "has_invalid_values": bool(n_nan > 0 or n_inf > 0),
        })
        
        # User diversity (if available)
        user_ids = [sample.get("user_id", "unknown") for sample in buffer_samples if "reward_vector" in sample]
        unique_users = len(set(user_ids))
        stats["unique_users"] = unique_users
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert replay buffer to format for gumbel.py analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Path to input replay buffer pickle file"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        required=True,
        help="Path to output .npz file for gumbel.py"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only analyze buffer without converting (dry run)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Load replay buffer
        logging.info(f"Loading replay buffer from: {args.input}")
        buffer_data = load_replay_buffer(args.input)
        
        # Extract reward matrix
        logging.info("Extracting reward vectors...")
        reward_matrix = extract_reward_matrix(buffer_data)
        
        # Analyze data
        stats = analyze_buffer_data(buffer_data, reward_matrix)
        
        # Print analysis
        print("\n" + "="*60)
        print("BUFFER ANALYSIS")
        print("="*60)
        print(f"Buffer size:           {stats['buffer_size']} samples")
        print(f"Buffer max capacity:   {stats['buffer_maxlen']}")
        print(f"Total samples collected: {stats['total_samples_collected']}")
        print(f"Batch count:           {stats['batch_count']}")
        print(f"Unique users:          {stats['unique_users']}")
        print()
        print(f"Reward matrix shape:   {stats['reward_matrix_shape']}")
        print(f"Number of samples:     {stats['n_samples']}")
        print(f"Number of attributes:  {stats['n_attributes']}")
        print(f"Reward statistics:")
        print(f"  Mean:                {stats['reward_mean']:.4f}")
        print(f"  Std:                 {stats['reward_std']:.4f}")
        print(f"  Min:                 {stats['reward_min']:.4f}")
        print(f"  Max:                 {stats['reward_max']:.4f}")
        
        if stats['has_invalid_values']:
            print(f"\n⚠️  WARNING: Found invalid values!")
            print(f"  NaN values:          {stats['n_nan_values']}")
            print(f"  Inf values:          {stats['n_inf_values']}")
        
        if args.check_only:
            print(f"\n✅ DRY RUN: Analysis complete. No files written.")
            return
        
        # Save converted data
        logging.info(f"Saving converted data to: {args.output}")
        save_for_gumbel_analysis(reward_matrix, args.output)
        
        print(f"\n✅ SUCCESS: Converted {stats['n_samples']} samples with {stats['n_attributes']} attributes")
        print(f"Output saved to: {args.output}")
        print(f"\nYou can now run gumbel analysis:")
        print(f"cd scripts/analysis && python gumbel.py")
        print(f"Make sure the script loads from: {args.output}")
        
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        print(f"\n❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()