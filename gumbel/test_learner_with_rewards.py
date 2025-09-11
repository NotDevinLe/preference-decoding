#!/usr/bin/env python3
"""
Test learner with precomputed rewards to verify training and attribute survival.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from skeleton import SparseMaskModel
from utils import bernoulli_gumbel_soft
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_precomputed_reward_data(reward_path="../data/reward_matrix_flexible.npz", num_batches=50, batch_size=32):
    """Load precomputed reward matrix and create batches"""
    
    logging.info(f"Loading precomputed rewards from {reward_path}")
    
    try:
        reward_data = np.load(reward_path)
        # Use Y_chosen like in the original gumbel.py
        X = reward_data['Y_chosen']  # [num_samples, d] - each row is one preference pair
        logging.info(f"Loaded Y_chosen shape: {X.shape}")
        
        num_samples, d = X.shape
        
        all_rewards = []
        
        for batch_idx in range(num_batches):
            # Sample batch_size rows from X (like original gumbel.py)
            sample_indices = np.random.choice(num_samples, size=batch_size, replace=True)
            R = torch.tensor(X[sample_indices], dtype=torch.float32)  # [batch_size, d]
            
            all_rewards.append(R)
            # No masks needed - we'll let the model learn its own mask
        
        # Analyze X to find potentially good attributes
        mean_rewards = np.mean(X, axis=0)
        std_rewards = np.std(X, axis=0)
        
        # Attributes with high mean rewards might be "good"
        good_threshold = np.mean(mean_rewards) + 0.5 * np.std(mean_rewards)
        potentially_good = np.where(mean_rewards > good_threshold)[0].tolist()
        
        logging.info(f"Reward statistics: mean={np.mean(mean_rewards):.4f}, std={np.std(mean_rewards):.4f}")
        logging.info(f"Potentially good attributes (mean > {good_threshold:.4f}): {potentially_good[:10]}...")
        
        return all_rewards, potentially_good, X.shape
        
    except FileNotFoundError:
        logging.error(f"Reward matrix file not found: {reward_path}")
        logging.info("Creating synthetic data instead...")
        return create_synthetic_reward_data(d=1000, num_batches=num_batches, batch_size=batch_size)
    except Exception as e:
        logging.error(f"Error loading reward matrix: {e}")
        logging.info("Creating synthetic data instead...")
        return create_synthetic_reward_data(d=1000, num_batches=num_batches, batch_size=batch_size)

def create_synthetic_reward_data(d=1000, num_batches=50, batch_size=32):
    """Fallback: Create synthetic reward data"""
    logging.info("Creating synthetic reward data...")
    
    # Create ground truth: some randomly selected attributes are "good"
    good_attributes = sorted(np.random.choice(d, size=min(20, d//10), replace=False).tolist())
    
    all_rewards = []
    
    for _ in range(num_batches):
        # Create reward matrix [batch_size, d]
        R = torch.randn(batch_size, d) * 0.1  # Base noise
        
        # Good attributes get higher rewards
        for attr_idx in good_attributes:
            R[:, attr_idx] += torch.normal(0.3, 0.1, (batch_size,))
        
        all_rewards.append(R)
    
    return all_rewards, good_attributes, (batch_size * num_batches, d)

def train_learner(reward_path="../data/reward_matrix_flexible.npz", k=20, num_steps=200, lr=0.01):
    """Train the learner model and return survived attributes"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device}")
    
    # Load precomputed reward data
    logging.info("Loading precomputed reward data...")
    all_rewards, good_attributes, data_shape = load_precomputed_reward_data(reward_path, num_steps)
    
    # Get dimensions from data
    d = data_shape[1]  # Number of attributes
    logging.info(f"Using d={d} attributes from reward matrix")
    logging.info(f"Potentially good attributes: {good_attributes[:10]}..." if len(good_attributes) > 10 else f"Potentially good attributes: {good_attributes}")
    
    # Create model
    model = SparseMaskModel(d, k, sparsity_weight=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    tau = 1.0
    losses = []
    reward_signals = []
    sparsity_levels = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Get batch data
        x = all_rewards[step].to(device)  # [batch_size, d] - treat rewards as input data
        
        # Normalize input like original gumbel.py
        x = F.normalize(x, p=2, dim=1)  # L2 normalize each sample
        
        # Forward pass - use the model's built-in forward method
        z, x_hat, masks = model.forward(x)
        
        # Reconstruction loss (exactly like original gumbel.py)
        recon_loss = F.mse_loss(x_hat, x)
        
        # Sparsity loss - encourage masks to be sparse (exactly like original)
        mask_probs = torch.sigmoid(model.mask_logits)
        sparsity_loss = mask_probs.mean()  # Penalize high probabilities
        
        # Total loss (exactly like original gumbel.py)
        loss = recon_loss + model.sparsity_weight * sparsity_loss
        
        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        losses.append(loss.item())
        reward_signals.append(recon_loss.item())  # Track reconstruction as reward signal
        sparsity_levels.append(mask_probs.sum().item())
        
        # Temperature annealing
        if step % 50 == 0 and step > 0:
            tau = max(0.1, tau * 0.95)
        
        if step % 20 == 0:
            # Check gradient flow
            grad_norm = model.mask_logits.grad.norm().item() if model.mask_logits.grad is not None else 0.0
            logits_std = model.mask_logits.std().item()
            
            logging.info(f"Step {step}: Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, "
                        f"Sparsity={mask_probs.sum().item():.1f}/{d}, Tau={tau:.3f}, "
                        f"GradNorm={grad_norm:.6f}, LogitsStd={logits_std:.6f}")
    
    # Final analysis
    final_probs = torch.sigmoid(model.mask_logits).detach().cpu()
    survived_attributes = torch.where(final_probs > 0.5)[0].tolist()
    
    logging.info("\n=== FINAL RESULTS ===")
    logging.info(f"Ground truth good attributes: {good_attributes}")
    logging.info(f"Survived attributes (prob > 0.5): {survived_attributes}")
    
    # Check overlap
    correct_survived = [attr for attr in survived_attributes if attr in good_attributes]
    false_positives = [attr for attr in survived_attributes if attr not in good_attributes]
    missed_good = [attr for attr in good_attributes if attr not in survived_attributes]
    
    logging.info(f"Correctly identified: {correct_survived} ({len(correct_survived)}/{len(good_attributes)})")
    logging.info(f"False positives: {false_positives} ({len(false_positives)})")
    logging.info(f"Missed good attributes: {missed_good} ({len(missed_good)})")
    
    precision = len(correct_survived) / len(survived_attributes) if survived_attributes else 0
    recall = len(correct_survived) / len(good_attributes)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logging.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Check if all probabilities are the same
    unique_probs = torch.unique(final_probs)
    logging.info(f"\nUnique probability values: {len(unique_probs)}")
    if len(unique_probs) <= 5:
        logging.info(f"Unique probabilities: {unique_probs.tolist()}")
    else:
        logging.info(f"Sample probabilities: {unique_probs[:10].tolist()}...")
    
    # Check probability statistics
    logging.info(f"Probability stats: min={final_probs.min():.6f}, max={final_probs.max():.6f}, "
                f"mean={final_probs.mean():.6f}, std={final_probs.std():.6f}")
    
    # Show top attributes by probability
    top_indices = torch.argsort(final_probs, descending=True)[:20]
    logging.info("\nTop 20 attributes by probability:")
    for i, idx in enumerate(top_indices):
        prob = final_probs[idx].item()
        is_good = "✅" if idx.item() in good_attributes else "❌"
        logging.info(f"  {i+1:2d}. Attr {idx.item():3d}: {prob:.6f} {is_good}")
    
    # Check if model parameters are actually changing
    logging.info(f"\nModel mask_logits stats:")
    logits = model.mask_logits.detach().cpu()
    logging.info(f"  Min: {logits.min():.6f}, Max: {logits.max():.6f}")
    logging.info(f"  Mean: {logits.mean():.6f}, Std: {logits.std():.6f}")
    logging.info(f"  Unique values: {len(torch.unique(logits))}")
    
    # Check gradient flow during training
    if hasattr(model.mask_logits, 'grad') and model.mask_logits.grad is not None:
        grad = model.mask_logits.grad
        logging.info(f"  Last gradient - Min: {grad.min():.6f}, Max: {grad.max():.6f}, Norm: {grad.norm():.6f}")
    else:
        logging.info("  No gradients found on mask_logits")
    
    return {
        'survived_attributes': survived_attributes,
        'good_attributes': good_attributes,
        'final_probs': final_probs.tolist(),
        'losses': losses,
        'reward_signals': reward_signals,
        'sparsity_levels': sparsity_levels,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_results_plot(results, output_path="training_results.png"):
    """Save training curves plot"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        ax1.plot(results['losses'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        
        # Reward signal
        ax2.plot(results['reward_signals'])
        ax2.set_title('Reward Signal')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        
        # Sparsity level
        ax3.plot(results['sparsity_levels'])
        ax3.set_title('Active Attributes')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Count')
        
        # Final probabilities
        ax4.hist(results['final_probs'], bins=50, alpha=0.7)
        ax4.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax4.set_title('Final Attribute Probabilities')
        ax4.set_xlabel('Probability')
        ax4.set_ylabel('Count')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved training plots to {output_path}")
        
    except ImportError:
        logging.warning("Matplotlib not available, skipping plot")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Learner with Precomputed Rewards")
    parser.add_argument("--reward-path", type=str, default="../data/reward_matrix_flexible.npz", 
                       help="Path to precomputed reward matrix")
    parser.add_argument("--k", type=int, default=20, help="Number of components")
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--plot", action="store_true", help="Save training plots")
    
    args = parser.parse_args()
    
    logging.info("=== Testing Learner with Precomputed Rewards ===")
    logging.info(f"Reward matrix: {args.reward_path}")
    logging.info(f"Model: ? attributes → {args.k} components (d will be determined from data)")
    logging.info(f"Training: {args.steps} steps with lr={args.lr}")
    
    # Run training
    results = train_learner(args.reward_path, args.k, args.steps, args.lr)
    
    # Save plots if requested
    if args.plot:
        save_results_plot(results)
    
    # Summary
    d_actual = len(results['final_probs'])
    logging.info("\n=== SUMMARY ===")
    logging.info(f"Training completed with F1 score: {results['f1']:.3f}")
    logging.info(f"Survived {len(results['survived_attributes'])} out of {d_actual} attributes")
    logging.info(f"Correctly identified {len([a for a in results['survived_attributes'] if a in results['good_attributes']])} out of {len(results['good_attributes'])} potentially good attributes")
    
    if results['f1'] > 0.5:
        logging.info("✅ Training successful - model learned to identify good attributes!")
    else:
        logging.info("❌ Training suboptimal - model struggled to identify good attributes")
    
    return 0

if __name__ == "__main__":
    main()