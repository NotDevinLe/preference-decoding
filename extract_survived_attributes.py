#!/usr/bin/env python3
"""
Script to extract survived/active attributes from trained preference decoding models.
This script demonstrates how to get the final set of attributes that survived training.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_survived_attributes_from_checkpoint(checkpoint_path: str, threshold: float = 0.5) -> Dict:
    """
    Extract survived attributes from a saved SparseMaskModel checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint (.pt file)
        threshold: Probability threshold for considering an attribute as "survived"
        
    Returns:
        Dictionary containing survived attributes and statistics
    """
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model state dict
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint
        
        # Get mask logits
        if "mask_logits" in model_state:
            mask_logits = model_state["mask_logits"]
        else:
            raise KeyError("mask_logits not found in checkpoint")
        
        # Convert to probabilities using sigmoid
        mask_probs = torch.sigmoid(mask_logits)
        
        # Find survived attributes (probability > threshold)
        survived_indices = torch.where(mask_probs > threshold)[0].tolist()
        
        # Get statistics
        stats = {
            "total_attributes": len(mask_logits),
            "survived_attributes": survived_indices,
            "num_survived": len(survived_indices),
            "survival_rate": len(survived_indices) / len(mask_logits),
            "threshold": threshold,
            "mask_probabilities": mask_probs.tolist(),
            "checkpoint_step": checkpoint.get("step", "unknown"),
            "model_config": checkpoint.get("model_config", {})
        }
        
        logging.info(f"Found {len(survived_indices)} survived attributes out of {len(mask_logits)} total")
        logging.info(f"Survival rate: {stats['survival_rate']:.3f}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise

def extract_survived_attributes_from_sparse_mask_model(model, threshold: float = 0.5) -> Dict:
    """
    Extract survived attributes from a trained SparseMaskModel instance.
    
    Args:
        model: Trained SparseMaskModel instance
        threshold: Probability threshold for considering an attribute as "survived"
        
    Returns:
        Dictionary containing survived attributes and statistics
    """
    if not hasattr(model, 'mask_logits'):
        raise ValueError("Model does not have mask_logits attribute")
    
    # Get mask probabilities
    mask_probs = torch.sigmoid(model.mask_logits).detach().cpu()
    
    # Find survived attributes
    survived_indices = torch.where(mask_probs > threshold)[0].tolist()
    
    stats = {
        "total_attributes": len(model.mask_logits),
        "survived_attributes": survived_indices,
        "num_survived": len(survived_indices),
        "survival_rate": len(survived_indices) / len(model.mask_logits),
        "threshold": threshold,
        "mask_probabilities": mask_probs.tolist()
    }
    
    return stats

def extract_survived_attributes_from_learner_server(server_url: str = "http://localhost:8002", threshold: float = 0.5) -> Dict:
    """
    Extract survived attributes from a running learner server.
    
    Args:
        server_url: URL of the learner server
        threshold: Probability threshold for considering an attribute as "survived"
        
    Returns:
        Dictionary containing survived attributes and statistics
    """
    import requests
    
    try:
        # Get parameters from learner server
        response = requests.get(f"{server_url}/parameters")
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("success", False):
            raise ValueError(f"Server error: {data.get('error', 'Unknown error')}")
        
        # Get mask logits
        mask_logits = torch.tensor(data["mask_logits"])
        
        # Convert to probabilities
        mask_probs = torch.sigmoid(mask_logits)
        
        # Find survived attributes
        survived_indices = torch.where(mask_probs > threshold)[0].tolist()
        
        stats = {
            "total_attributes": len(mask_logits),
            "survived_attributes": survived_indices,
            "num_survived": len(survived_indices),
            "survival_rate": len(survived_indices) / len(mask_logits),
            "threshold": threshold,
            "mask_probabilities": mask_probs.tolist(),
            "server_step": data["step"],
            "server_tau": data["tau"]
        }
        
        logging.info(f"Retrieved from server at step {data['step']}")
        logging.info(f"Found {len(survived_indices)} survived attributes out of {len(mask_logits)} total")
        
        return stats
        
    except requests.RequestException as e:
        logging.error(f"Error connecting to server: {e}")
        raise
    except Exception as e:
        logging.error(f"Error extracting from server: {e}")
        raise

def analyze_attribute_survival(stats: Dict, top_k: int = 20) -> None:
    """
    Analyze and print detailed survival statistics.
    
    Args:
        stats: Statistics dictionary from extract_survived_attributes_*
        top_k: Number of top attributes to display
    """
    print("\n" + "="*60)
    print("ATTRIBUTE SURVIVAL ANALYSIS")
    print("="*60)
    
    print(f"Total attributes: {stats['total_attributes']}")
    print(f"Survived attributes: {stats['num_survived']}")
    print(f"Survival rate: {stats['survival_rate']:.1%}")
    print(f"Threshold: {stats['threshold']}")
    
    if "checkpoint_step" in stats:
        print(f"Training step: {stats['checkpoint_step']}")
    if "server_step" in stats:
        print(f"Server step: {stats['server_step']}")
    
    # Show survived attribute indices
    survived = stats['survived_attributes']
    if survived:
        print(f"\nSurvived attribute indices: {survived[:20]}{'...' if len(survived) > 20 else ''}")
    else:
        print("\nNo attributes survived!")
    
    # Show top attributes by probability
    probs = stats['mask_probabilities']
    if probs:
        prob_tensor = torch.tensor(probs)
        top_indices = torch.argsort(prob_tensor, descending=True)[:top_k]
        
        print(f"\nTop {min(top_k, len(probs))} attributes by probability:")
        print("-" * 40)
        for i, idx in enumerate(top_indices):
            prob = prob_tensor[idx].item()
            survived_mark = "✓" if idx.item() in survived else "✗"
            print(f"{i+1:2d}. Attribute {idx.item():3d}: {prob:.6f} {survived_mark}")
    
    # Show probability distribution
    if probs:
        prob_array = np.array(probs)
        print(f"\nProbability distribution:")
        print(f"  Min: {prob_array.min():.6f}")
        print(f"  Max: {prob_array.max():.6f}")
        print(f"  Mean: {prob_array.mean():.6f}")
        print(f"  Std: {prob_array.std():.6f}")
        print(f"  Median: {np.median(prob_array):.6f}")

def save_survived_attributes(stats: Dict, output_path: str) -> None:
    """
    Save survived attributes to a JSON file.
    
    Args:
        stats: Statistics dictionary from extract_survived_attributes_*
        output_path: Path to save the JSON file
    """
    # Create a clean output dictionary
    output_data = {
        "survived_attributes": stats["survived_attributes"],
        "num_survived": stats["num_survived"],
        "total_attributes": stats["total_attributes"],
        "survival_rate": stats["survival_rate"],
        "threshold": stats["threshold"],
        "mask_probabilities": stats["mask_probabilities"]
    }
    
    # Add training info if available
    if "checkpoint_step" in stats:
        output_data["training_step"] = stats["checkpoint_step"]
    if "server_step" in stats:
        output_data["server_step"] = stats["server_step"]
    if "model_config" in stats:
        output_data["model_config"] = stats["model_config"]
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Saved survived attributes to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract survived attributes from preference decoding models")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint file")
    parser.add_argument("--server-url", type=str, default="http://localhost:8002", 
                       help="URL of learner server")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Probability threshold for survived attributes")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--top-k", type=int, default=20, 
                       help="Number of top attributes to display")
    parser.add_argument("--source", choices=["checkpoint", "server"], default="checkpoint",
                       help="Source to extract attributes from")
    
    args = parser.parse_args()
    
    try:
        if args.source == "checkpoint":
            if not args.checkpoint:
                print("Error: --checkpoint is required when source is 'checkpoint'")
                return
            if not Path(args.checkpoint).exists():
                print(f"Error: Checkpoint file not found: {args.checkpoint}")
                return
            
            stats = extract_survived_attributes_from_checkpoint(args.checkpoint, args.threshold)
            
        elif args.source == "server":
            stats = extract_survived_attributes_from_learner_server(args.server_url, args.threshold)
        
        # Analyze and display results
        analyze_attribute_survival(stats, args.top_k)
        
        # Save if requested
        if args.output:
            save_survived_attributes(stats, args.output)
            
    except Exception as e:
        logging.error(f"Failed to extract survived attributes: {e}")
        return

if __name__ == "__main__":
    main()