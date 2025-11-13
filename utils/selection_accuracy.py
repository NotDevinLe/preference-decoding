import torch
import os
import glob
import random
import numpy as np
import math


def compute_normalized_log_probs(rewards):
    """
    Compute normalized log probabilities following the derived formulation.
    """
    eps = 1e-12
    attr_log_probs = (rewards['attr_scores_chosen'] / 
                      rewards['attr_counts_chosen'].clamp(min=eps))
    
    base_log_probs = (rewards['base_scores_chosen'].unsqueeze(1) / 
                      rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps))
    
    attr_log_probs_rejected = (rewards['attr_scores_rejected'] / 
                      rewards['attr_counts_rejected'].clamp(min=eps))

    base_log_probs_rejected = (rewards['base_scores_rejected'].unsqueeze(1) / 
                      rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps))
    
    # (chosen_attr - chosen_base) - (rejected_attr - rejected_base)
    return (attr_log_probs - chosen_base_avg) - (attr_log_probs_rejected - rejected_base_avg)


def l1_solve(d_mean, l1_lambda, std=None):
    """
    Closed-form solution to: maximize d^T p - lambda * ||p||_1  s.t. ||p||_2 <= 1
    """
    d = np.asarray(d_mean, dtype=float).copy()
    # soft-threshold
    z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
    norm = np.linalg.norm(z, ord=2)
    if norm == 0.0:
        return np.zeros_like(d)
    if std is None:
        return (z / norm).copy()
    else:
        return (z / (norm * std)).copy()


def load_rewards_dir(rewards_dir="eval_rewards/llama8b/persona/train", num_users=None):
    """
    Load reward files without modification.
    """
    pattern = "user*.pt"
    reward_files = glob.glob(os.path.join(rewards_dir, pattern))
    reward_files.sort()

    if len(reward_files) == 0:
        print(f"No reward files found in {rewards_dir}")
        return {}

    if num_users is not None:
        reward_files = reward_files[:num_users]
    print(f"Loading {len(reward_files)} reward files from {rewards_dir}\n")

    rewards_by_user = {}
    for reward_file in reward_files:
        user_id = os.path.basename(reward_file).replace('.pt', '')
        rewards = torch.load(reward_file, map_location="cpu")
        rewards_by_user[user_id] = rewards
        print(f"Loaded {user_id}: {rewards['attr_scores_chosen'].shape[0]} samples")

    return rewards_by_user


def load_and_modify_rewards(rewards_dir="eval_rewards/llama8b/persona/train", num_users=15):
    """
    Load reward files and create modified versions by replacing rejected answers 
    with randomly sampled chosen answers from other users for the same question.
    This happens in memory without modifying any files.
    
    Args:
        rewards_dir: Directory containing the reward files
        num_users: Number of users to process (default: 15)
    
    Returns:
        Dictionary mapping user_id to modified rewards
    """
    # Find all reward files
    pattern = "user*.pt"
    reward_files = glob.glob(os.path.join(rewards_dir, pattern))
    reward_files.sort()
    
    if len(reward_files) == 0:
        print(f"No reward files found in {rewards_dir}")
        return {}
    
    # Take only the first num_users files
    reward_files = reward_files[:num_users]
    print(f"Processing first {len(reward_files)} reward files from {rewards_dir}\n")
    
    # Load all reward data
    all_rewards = {}
    user_ids = []
    
    for reward_file in reward_files:
        user_id = os.path.basename(reward_file).replace('.pt', '')
        user_ids.append(user_id)
        rewards = torch.load(reward_file, map_location="cpu")
        all_rewards[user_id] = rewards
        print(f"Loaded {user_id}: {rewards['attr_scores_chosen'].shape[0]} samples")
    
    print(f"\nProcessing {len(user_ids)} users...")
    
    # Create modified rewards for each user
    modified_rewards = {}
    
    for user_id in user_ids:
        print(f"\nProcessing {user_id}...")
        
        user_rewards = all_rewards[user_id]
        n_samples = user_rewards['attr_scores_chosen'].shape[0]
        
        # Get other users (exclude current user)
        other_users = [uid for uid in user_ids if uid != user_id]
        
        # Create modified rewards by copying the original
        # Handle both tensors and nested dicts
        modified = {}
        for key in user_rewards.keys():
            if isinstance(user_rewards[key], dict):
                # For nested dictionaries (like 'metadata'), recursively clone
                modified[key] = user_rewards[key].copy()
            elif hasattr(user_rewards[key], 'clone'):
                # For tensors, use clone
                modified[key] = user_rewards[key].clone()
            else:
                # For other types, just copy
                modified[key] = user_rewards[key]
        
        for sample_idx in range(n_samples):
            sampled_user = random.choice(other_users)
            sampled_rewards = all_rewards[sampled_user]
            
            # Ensure the sampled user has this sample index
            if sample_idx < sampled_rewards['attr_scores_chosen'].shape[0]:
                # Replace rejected scores with chosen scores from sampled user
                modified['attr_scores_rejected'][sample_idx] = sampled_rewards['attr_scores_chosen'][sample_idx]
                modified['attr_counts_rejected'][sample_idx] = sampled_rewards['attr_counts_chosen'][sample_idx]
                modified['base_scores_rejected'][sample_idx] = sampled_rewards['base_scores_chosen'][sample_idx]
                modified['base_counts_rejected'][sample_idx] = sampled_rewards['base_counts_chosen'][sample_idx]
            else:
                print(f"  Warning: Sampled user {sampled_user} doesn't have sample {sample_idx}, skipping")
        
        modified_rewards[user_id] = modified
        print(f"  Created modified rewards for {user_id}")
    
    print(f"\nCompleted processing {len(user_ids)} users")
    return modified_rewards


def approximate_local(modified_rewards, test_user_id, train_ratio=0.8, l1_lambda=0.01, selected_attributes=None, leave_one_out=False):
    """
    Local approximation using modified rewards data.
    
    Args:
        modified_rewards: Dictionary of modified reward data
        test_user_id: ID of the test user
        train_ratio: Ratio of data to use for training (ignored if leave_one_out=True)
        l1_lambda: L1 regularization parameter
        selected_attributes: List of attribute indices to use (e.g., [381, 83]). 
                            If None, uses all attributes.
        leave_one_out: If True, perform leave-one-out cross-validation
    
    Returns:
        If leave_one_out=False: Tuple of (p vector, train_indices, test_indices, selected_attributes)
        If leave_one_out=True: Tuple of (all_ps list, all_train_indices, all_test_indices, selected_attributes)
                               where all_ps is a list of p vectors (one per fold),
                               and all_train_indices and all_test_indices are lists of lists
    """
    if test_user_id not in modified_rewards:
        raise ValueError(f"Test user {test_user_id} not found in modified rewards")
    
    test_rewards = modified_rewards[test_user_id]
    n_samples, k_attrs = test_rewards['attr_scores_chosen'].shape
    
    if selected_attributes is not None:
        if isinstance(selected_attributes, list):
            selected_attributes = torch.tensor(selected_attributes, dtype=torch.long)
        print(f"Using {len(selected_attributes)} selected attributes: {selected_attributes.tolist()}")
    else:
        selected_attributes = torch.arange(k_attrs)
        print(f"Using all {k_attrs} attributes")
    
    eps = 1e-12
    
    chosen_attr_probs = test_rewards['attr_scores_chosen'] / test_rewards['attr_counts_chosen'].clamp(min=eps)
    chosen_base_probs = test_rewards['base_scores_chosen'].unsqueeze(1) / test_rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
    chosen_log_probs = chosen_attr_probs - chosen_base_probs
    
    rejected_attr_probs = test_rewards['attr_scores_rejected'] / test_rewards['attr_counts_rejected'].clamp(min=eps)
    rejected_base_probs = test_rewards['base_scores_rejected'].unsqueeze(1) / test_rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
    rejected_log_probs = rejected_attr_probs - rejected_base_probs
    
    chosen_log_probs = chosen_log_probs[:, selected_attributes].contiguous()
    rejected_log_probs = rejected_log_probs[:, selected_attributes].contiguous()
    
    # Build X matrix: (chosen_attr - chosen_base) - (rejected_attr - rejected_base)
    X = chosen_log_probs - rejected_log_probs
    
    if leave_one_out:
        # Leave-one-out cross-validation: for each sample, train on all others and test on that one
        print(f"Performing leave-one-out evaluation on {n_samples} samples...")
        all_ps = []
        all_train_indices = []
        all_test_indices = []
        
        for test_idx in range(n_samples):
            # Create train indices (all indices except test_idx)
            train_indices = [i for i in range(n_samples) if i != test_idx]
            test_indices = [test_idx]
            
            X_train = X[train_indices]
            
            col_std = X_train.std(dim=0).clamp_min(1e-8)
            X_standardized = X_train / col_std
            d = X_standardized.mean(dim=0).detach().cpu().numpy().copy()
            
            # Solve for p vector with L1 regularization
            col_std_np = col_std.detach().cpu().numpy().copy()
            p = l1_solve(d, l1_lambda, std=col_std_np)
            
            all_ps.append(p)
            all_train_indices.append(train_indices)
            all_test_indices.append(test_indices)
            
            if (test_idx + 1) % max(1, n_samples // 10) == 0:
                print(f"  Processed {test_idx + 1}/{n_samples} folds...")
        
        # Compute average p vector for display purposes
        avg_p = np.mean(all_ps, axis=0)
        print(f"Computed {len(all_ps)} p vectors (one per fold)")
        print(f"Average p vector has {np.count_nonzero(avg_p)} non-zero weights")
        if np.count_nonzero(avg_p) > 0:
            top_k = min(5, len(avg_p))
            top_indices = np.argsort(np.abs(avg_p))[-top_k:][::-1].copy()
            print(f"Top {top_k} attribute indices (in selected set): {top_indices.tolist()}")
            if selected_attributes is not None:
                print(f"Top {top_k} absolute attribute indices: {selected_attributes[top_indices].tolist()}")
        
        return all_ps, all_train_indices, all_test_indices, selected_attributes
    else:
        # Standard train/test split
        n_train = int(n_samples * train_ratio)
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_samples))
        
        print(f"Training on {n_train} samples, testing on {len(test_indices)} samples")
        
        X_train = X[train_indices]
        
        col_std = X_train.std(dim=0).clamp_min(1e-8)
        X_standardized = X_train / col_std
        d = X_standardized.mean(dim=0).detach().cpu().numpy().copy()
        
        # Solve for p vector with L1 regularization
        col_std_np = col_std.detach().cpu().numpy().copy()
        p = l1_solve(d, l1_lambda, std=col_std_np)
        
        print(f"Computed p vector with {np.count_nonzero(p)} non-zero weights")
        if np.count_nonzero(p) > 0:
            top_k = min(5, len(p))
            top_indices = np.argsort(np.abs(p))[-top_k:][::-1].copy()  # copy() removes negative strides
            print(f"Top {top_k} attribute indices (in selected set): {top_indices.tolist()}")
            if selected_attributes is not None:
                print(f"Top {top_k} absolute attribute indices: {selected_attributes[top_indices].tolist()}")
        
        return p, train_indices, test_indices, selected_attributes


def evaluate_accuracy_local(modified_rewards, test_user_id, p, train_indices, test_indices, selected_attributes=None, leave_one_out=False):
    """
    Evaluate preference pair accuracy on test data using the learned p vector.
    Based on the async evaluate_accuracy function but using pre-computed reward matrices.
    
    Args:
        modified_rewards: Dictionary of modified reward data
        test_user_id: ID of the test user
        p: learned drift vector (or average p vector if leave_one_out=True)
        train_indices: Training sample indices (or list of lists if leave_one_out=True)
        test_indices: Test sample indices (or list of lists if leave_one_out=True)
        selected_attributes: List or tensor of attribute indices used during training
        leave_one_out: If True, evaluate using leave-one-out folds
    
    Returns:
        accuracy (float) - average accuracy across all folds if leave_one_out=True
    """
    if test_user_id not in modified_rewards:
        raise ValueError(f"Test user {test_user_id} not found in modified rewards")
    
    test_rewards = modified_rewards[test_user_id]
    eps = 1e-12
    
    # Compute normalized log probabilities
    chosen_attr_probs = test_rewards['attr_scores_chosen'] / test_rewards['attr_counts_chosen'].clamp(min=eps)
    chosen_base_probs = test_rewards['base_scores_chosen'].unsqueeze(1) / test_rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
    chosen_log_probs = chosen_attr_probs - chosen_base_probs
    
    rejected_attr_probs = test_rewards['attr_scores_rejected'] / test_rewards['attr_counts_rejected'].clamp(min=eps)
    rejected_base_probs = test_rewards['base_scores_rejected'].unsqueeze(1) / test_rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
    rejected_log_probs = rejected_attr_probs - rejected_base_probs
    
    # Filter to selected attributes if specified (ensure contiguous tensors)
    if selected_attributes is not None:
        if isinstance(selected_attributes, list):
            selected_attributes = torch.tensor(selected_attributes, dtype=torch.long)
        chosen_log_probs = chosen_log_probs[:, selected_attributes].contiguous()
        rejected_log_probs = rejected_log_probs[:, selected_attributes].contiguous()
    
    if leave_one_out:
        # Evaluate each fold separately using its own p vector
        all_accuracies = []
        
        for fold_idx, (p_fold, train_idx_fold, test_idx_fold) in enumerate(zip(p, train_indices, test_indices)):
            # Use this fold's p vector
            p_tensor = torch.tensor(p_fold.copy(), dtype=torch.float32)
            
            # Compute drift scores for this fold's test data
            X_test = chosen_log_probs[test_idx_fold] - rejected_log_probs[test_idx_fold]
            drift_scores = X_test @ p_tensor
            
            # Count correct predictions (positive drift means chosen > rejected)
            correct = (drift_scores > 0).sum().item()
            accuracy = correct / len(test_idx_fold)
            all_accuracies.append(accuracy)
        
        avg_accuracy = np.mean(all_accuracies)
        print(f"Leave-one-out accuracy: {avg_accuracy:.4f} ({sum(all_accuracies):.0f}/{len(all_accuracies)})")
        print(f"  Accuracy range: [{min(all_accuracies):.4f}, {max(all_accuracies):.4f}]")
        
        return avg_accuracy
    else:
        # Standard evaluation
        # Compute drift scores for test data
        X_test = chosen_log_probs[test_indices] - rejected_log_probs[test_indices]
        drift_scores = X_test @ torch.tensor(p.copy(), dtype=torch.float32)
        
        # Count correct predictions (positive drift means chosen > rejected)
        correct = (drift_scores > 0).sum().item()
        accuracy = correct / len(test_indices)
        
        print(f"Test accuracy: {accuracy:.4f} ({correct}/{len(test_indices)})")
        print(f"Mean drift score: {drift_scores.mean().item():.4f}")
        
        return accuracy

def compute_average_metrics(modified_rewards, selected_attributes, l1_lambda):
    """
    Compute average test accuracy and average number of non-zero weights across users
    for a given attribute set and l1_lambda.
    Returns (avg_accuracy, avg_nonzero, num_users).
    """
    all_accuracies = []
    all_nonzero = []
    for test_user_id in modified_rewards.keys():
        try:
            p, train_indices, test_indices, selected_attrs = approximate_local(
                modified_rewards,
                test_user_id,
                train_ratio=0.8,
                l1_lambda=l1_lambda,
                selected_attributes=selected_attributes,
            )
            accuracy = evaluate_accuracy_local(
                modified_rewards,
                test_user_id,
                p,
                train_indices,
                test_indices,
                selected_attributes=selected_attrs,
            )
            all_accuracies.append(accuracy)
            all_nonzero.append(int(np.count_nonzero(p)))
        except Exception:
            continue
    if not all_accuracies:
        return 0.0, 0.0, 0
    return float(np.mean(all_accuracies)), float(np.mean(all_nonzero)), len(all_accuracies)

def main(selected_attributes=None, l1_lambda=0.01, seed=0,
         mismatch_training=True, rewards_dir="eval_rewards/llama1b/prism/train",
         num_users=None, leave_one_out=False):
    """
    Main function to evaluate selection accuracy.
    
    Args:
        selected_attributes: List of attribute indices to use (e.g., [381, 83]).
                           If None, uses all attributes.
    """
    # Set seeds for determinism
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load rewards (with optional mismatching)
    if mismatch_training:
        print("Loading rewards with mismatched (shuffled) rejected answers for training.")
        modified_rewards = load_and_modify_rewards(rewards_dir, num_users if num_users is not None else 15)
    else:
        print("Loading rewards without modifying rejected answers.")
        modified_rewards = load_rewards_dir(rewards_dir, num_users)
    
    if not modified_rewards:
        print("No rewards loaded, exiting")
        return
    
    # Print selected attributes info
    if selected_attributes is not None:
        print(f"\nUsing {len(selected_attributes)} selected attributes: {selected_attributes}")
    else:
        print(f"\nUsing all available attributes")
    
    # Evaluate all users
    all_accuracies = []
    all_nonzero_weights = []
    
    for test_user_id in modified_rewards.keys():
        print(f"\n{'='*60}")
        print(f"Testing approximation on user {test_user_id}")
        print(f"{'='*60}")
        
        try:
            # Run approximation
            p, train_indices, test_indices, selected_attrs = approximate_local(
                modified_rewards, 
                test_user_id, 
                train_ratio=0.8, 
                l1_lambda=l1_lambda,
                selected_attributes=selected_attributes,
                leave_one_out=leave_one_out
            )
            
            # Evaluate accuracy
            accuracy = evaluate_accuracy_local(
                modified_rewards, 
                test_user_id, 
                p, 
                train_indices, 
                test_indices,
                selected_attributes=selected_attrs,
                leave_one_out=leave_one_out
            )
            
            all_accuracies.append(accuracy)
            
            # Handle non-zero weights calculation
            if leave_one_out:
                # For leave-one-out, compute average non-zero weights across all folds
                avg_nonzero = np.mean([np.count_nonzero(p_fold) for p_fold in p])
                avg_p = np.mean(p, axis=0)
                all_nonzero_weights.append(avg_nonzero)
                print(f"\nResults for {test_user_id}:")
                print(f"Leave-one-out Accuracy: {accuracy:.4f}")
                print(f"Average Non-zero weights: {avg_nonzero:.1f}/{len(avg_p)}")
                print("Learned weights p (average across folds, aligned to selected attributes):")
                print(np.array2string(avg_p, precision=6, suppress_small=True, max_line_width=120))
                # If a subset of attributes was used, print mapping of absolute idx -> weight for non-zero weights
                if selected_attrs is not None:
                    nz_idx = np.nonzero(avg_p)[0]
                    if nz_idx.size > 0:
                        abs_idx = selected_attrs[nz_idx].tolist()
                        weights = avg_p[nz_idx].tolist()
                        print("Non-zero weights (absolute_index: weight):")
                        print({int(i): float(w) for i, w in zip(abs_idx, weights)})
            else:
                all_nonzero_weights.append(np.count_nonzero(p))
                print(f"\nResults for {test_user_id}:")
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Non-zero weights: {np.count_nonzero(p)}/{len(p)}")
                print("Learned weights p (aligned to selected attributes):")
                print(np.array2string(p, precision=6, suppress_small=True, max_line_width=120))
                # If a subset of attributes was used, print mapping of absolute idx -> weight for non-zero weights
                if selected_attrs is not None:
                    nz_idx = np.nonzero(p)[0]
                    if nz_idx.size > 0:
                        abs_idx = selected_attrs[nz_idx].tolist()
                        weights = p[nz_idx].tolist()
                        print("Non-zero weights (absolute_index: weight):")
                        print({int(i): float(w) for i, w in zip(abs_idx, weights)})
            
        except Exception as e:
            print(f"Error processing {test_user_id}: {e}")
            continue
    
    # Compute and display average results
    if all_accuracies:
        avg_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        avg_nonzero = np.mean(all_nonzero_weights)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ACROSS ALL USERS")
        print(f"{'='*60}")
        eval_mode = "leave-one-out" if leave_one_out else "train/test split"
        if selected_attributes is not None:
            print(f"Selected attributes: {selected_attributes}")
        print(f"Evaluation mode: {eval_mode}")
        print(f"Average Test Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average Non-zero weights: {avg_nonzero:.1f}")
        print(f"Number of users evaluated: {len(all_accuracies)}")
        print(f"Accuracy range: [{min(all_accuracies):.4f}, {max(all_accuracies):.4f}]")
    else:
        print("No users were successfully evaluated")

attributes = {
    2: [381, 83],
    5: [362, 20, 337, 82, 277],
    10: [357, 15, 332, 14, 324, 28, 279, 17, 136, 293],
    20: [147, 215, 261, 71, 133, 376, 59, 332, 359, 152, 168, 330, 239, 49, 3, 207, 277, 205, 298, 16],
    30: [30, 140, 270, 260, 117, 153, 56, 393, 390, 380, 87, 187, 331, 298, 49, 207, 363, 198, 172, 110, 52, 59, 120, 74, 225, 293, 373, 83, 114, 370],
    40: [390, 128, 240, 34, 275, 198, 343, 74, 302, 49, 120, 35, 308, 392, 278, 131, 168, 270, 76, 187, 327, 27, 369, 199, 60, 20, 70, 209, 219, 155, 78, 117, 225, 260, 235, 136, 110, 265, 16, 395],
    50: [393, 262, 104, 74, 123, 66, 230, 83, 307, 207, 128, 390, 275, 293, 117, 68, 239, 136, 261, 359, 249, 285, 302, 70, 69, 119, 327, 323, 91, 3, 34, 399, 353, 369, 168, 277, 218, 176, 233, 199, 44, 175, 346, 227, 260, 75, 282, 225, 380, 254],
    60: [330, 238, 130, 81, 187, 332, 146, 185, 49, 341, 397, 95, 225, 62, 393, 56, 2, 260, 293, 249, 74, 128, 112, 178, 55, 371, 83, 256, 362, 136, 152, 15, 13, 311, 193, 69, 285, 117, 207, 377, 27, 337, 101, 110, 176, 119, 275, 87, 91, 75, 70, 77, 265, 309, 198, 168, 133, 53, 390, 135],
    120: [8, 387, 56, 196, 133, 2, 310, 126, 249, 73, 338, 95, 127, 218, 155, 232, 380, 9, 168, 190, 86, 393, 387, 66, 87, 122, 295, 346, 245, 396, 257, 83, 145, 243, 89, 170, 46, 343, 252, 120, 16, 153, 103, 363, 390, 78, 147, 119, 162, 90, 112, 260, 91, 399, 194, 273, 178, 54, 303, 327, 305, 161, 198, 13, 332, 227, 152, 48, 141, 375, 39, 121, 360, 114, 283, 100, 308, 397, 146, 304, 125, 49, 254, 262, 169, 369, 106, 58, 59, 223, 207, 176, 80, 330, 275, 158, 27, 302, 34, 3, 182, 5, 307, 188, 104, 225, 225, 355, 270, 319, 199, 29, 117, 364, 128, 134, 344, 144, 378, 277],
    180: [104, 29, 13, 38, 15, 2, 292, 136, 187, 284, 374, 249, 302, 340, 199, 178, 165, 219, 39, 90, 369, 287, 172, 34, 308, 281, 114, 265, 72, 77, 389, 144, 114, 176, 342, 349, 390, 262, 360, 91, 27, 62, 244, 142, 384, 123, 152, 341, 113, 296, 330, 162, 64, 224, 98, 278, 139, 49, 256, 147, 392, 236, 121, 120, 134, 300, 257, 186, 331, 377, 225, 288, 343, 285, 67, 20, 332, 225, 222, 364, 55, 295, 273, 338, 56, 399, 252, 115, 158, 130, 242, 275, 76, 188, 206, 92, 85, 175, 16, 48, 101, 168, 89, 131, 397, 198, 70, 140, 135, 35, 260, 316, 118, 386, 74, 27, 96, 232, 23, 254, 169, 63, 325, 57, 214, 276, 155, 52, 352, 339, 183, 60, 107, 99, 309, 243, 46, 148, 218, 350, 37, 91, 270, 393, 370, 45, 399, 10, 66, 73, 190, 59, 371, 355, 78, 128, 103, 266, 146, 213, 19, 387, 228, 333, 261, 145, 12, 327, 251, 95, 29, 307, 305, 83, 346, 361, 358, 319, 17, 30],
    240: [299, 122, 168, 165, 68, 385, 397, 17, 182, 389, 330, 316, 265, 99, 237, 91, 266, 222, 285, 193, 399, 112, 97, 179, 140, 20, 361, 30, 216, 91, 17, 59, 181, 249, 189, 139, 138, 152, 56, 291, 29, 364, 225, 127, 71, 300, 109, 302, 346, 272, 243, 61, 334, 309, 49, 219, 133, 319, 225, 146, 123, 103, 66, 2, 270, 390, 307, 74, 37, 75, 260, 151, 304, 144, 387, 82, 236, 106, 350, 114, 39, 207, 264, 81, 43, 72, 120, 203, 15, 257, 160, 356, 145, 262, 124, 167, 386, 58, 94, 136, 346, 396, 134, 27, 395, 208, 252, 155, 231, 355, 263, 344, 26, 95, 254, 351, 55, 35, 69, 159, 31, 174, 188, 322, 27, 274, 76, 190, 163, 162, 119, 352, 276, 238, 327, 298, 313, 121, 377, 343, 13, 172, 73, 64, 13, 170, 372, 244, 342, 19, 118, 44, 261, 57, 234, 301, 89, 310, 9, 132, 338, 87, 359, 198, 275, 176, 16, 27, 247, 135, 72, 306, 62, 42, 196, 223, 161, 371, 369, 47, 105, 9, 290, 363, 104, 325, 311, 374, 232, 102, 77, 23, 86, 153, 193, 278, 349, 270, 375, 48, 308, 146, 8, 295, 111, 256, 185, 336, 303, 178, 228, 162, 169, 6, 130, 187, 100, 208, 90, 245, 213, 360, 70, 78, 147, 294, 391, 212, 394, 380, 259, 11, 215, 253, 92, 201, 227, 357, 153, 285],
    26: list(range(26))
}

if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Evaluate selection accuracy without plotting.")
    parser.add_argument("--l1_lambda", type=float, default=0.01, help="L1 regularization strength (default: 0.01)")
    parser.add_argument("--k", type=int, required=True, help="Number of attributes to use (use preset indices)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--rewards_dir", type=str, default="eval_rewards/llama8b/persona/train",
                        help="Directory containing reward .pt files")
    parser.add_argument("--num_users", type=int, default=None,
                        help="Number of user files to load (default: all)")
    parser.add_argument("--mismatch_training", action="store_true",
                        help="If set, replace rejected responses with randomly selected chosen responses from other users")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="If set, perform leave-one-out cross-validation")
    args = parser.parse_args()

    # Choose attribute subset by preset mapping; if k not found, use all attributes
    selected_attributes = attributes.get(args.k, None)

    # Run a single evaluation pass (no plots)
    main(selected_attributes=selected_attributes,
         l1_lambda=args.l1_lambda,
         seed=args.seed,
         mismatch_training=args.mismatch_training,
         rewards_dir=args.rewards_dir,
         num_users=args.num_users,
         leave_one_out=args.leave_one_out)