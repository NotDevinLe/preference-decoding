import torch
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
# Set a clean default style once
plt.style.use('seaborn-v0_8-whitegrid')
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
    return (attr_log_probs - base_log_probs) - (attr_log_probs_rejected - base_log_probs_rejected)


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


def load_and_modify_rewards(rewards_dir="rewards_persona_testing", num_users=15):
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


def approximate_local(modified_rewards, test_user_id, train_ratio=0.8, l1_lambda=0.01, selected_attributes=None):
    """
    Local approximation using modified rewards data.
    
    Args:
        modified_rewards: Dictionary of modified reward data
        test_user_id: ID of the test user
        train_ratio: Ratio of data to use for training
        l1_lambda: L1 regularization parameter
        selected_attributes: List of attribute indices to use (e.g., [381, 83]). 
                            If None, uses all attributes.
    
    Returns:
        Tuple of (p vector, train_indices, test_indices, selected_attributes)
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
    
    n_train = int(n_samples * train_ratio)
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_samples))
    
    print(f"Training on {n_train} samples, testing on {len(test_indices)} samples")
    
    eps = 1e-12
    
    chosen_attr_probs = test_rewards['attr_scores_chosen'] / test_rewards['attr_counts_chosen'].clamp(min=eps)
    chosen_base_probs = test_rewards['base_scores_chosen'].unsqueeze(1) / test_rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
    chosen_log_probs = chosen_attr_probs - chosen_base_probs
    
    rejected_attr_probs = test_rewards['attr_scores_rejected'] / test_rewards['attr_counts_rejected'].clamp(min=eps)
    rejected_base_probs = test_rewards['base_scores_rejected'].unsqueeze(1) / test_rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
    rejected_log_probs = rejected_attr_probs - rejected_base_probs
    
    chosen_log_probs = chosen_log_probs[:, selected_attributes].contiguous()
    rejected_log_probs = rejected_log_probs[:, selected_attributes].contiguous()
    
        # Build X matrix for training data: (chosen_attr - chosen_base) - (rejected_attr - rejected_base)
    X_train = chosen_log_probs[train_indices] - rejected_log_probs[train_indices]
    
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


def evaluate_accuracy_local(modified_rewards, test_user_id, p, train_indices, test_indices, selected_attributes=None):
    """
    Evaluate preference pair accuracy on test data using the learned p vector.
    Based on the async evaluate_accuracy function but using pre-computed reward matrices.
    
    Args:
        modified_rewards: Dictionary of modified reward data
        test_user_id: ID of the test user
        p: learned drift vector
        train_indices: Training sample indices
        test_indices: Test sample indices
        selected_attributes: List or tensor of attribute indices used during training
    
    Returns:
        accuracy (float)
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


def sweep_lambdas_and_plot(modified_rewards, attributes_dict, lambdas, out_path, out_nonzero_path=None):
    """
    For each attribute count in attributes_dict, compute average accuracy across users for each lambda,
    and plot accuracy vs lambda (one line per attribute count). Also plots average number of non-zero
    weights vs lambda to a second file.
    """
    plt.figure(figsize=(8, 5))
    for k in sorted(attributes_dict.keys()):
        selected = attributes_dict[k]
        y = []
        nz = []
        for lam in lambdas:
            avg_acc, avg_nz, n_users = compute_average_metrics(modified_rewards, selected, lam)
            y.append(avg_acc)
            nz.append(avg_nz)
        plt.plot(lambdas, y, marker='o', label=f"k={k}")

    plt.xlabel("l1_lambda")
    plt.ylabel("Average test accuracy")
    plt.title("Average accuracy vs l1_lambda across attribute counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved lambda sweep plot to {out_path}")

    # Second figure: average number of non-zero weights vs lambda
    if out_nonzero_path is None:
        if out_path.lower().endswith('.png'):
            out_nonzero_path = out_path[:-4] + '_nonzero.png'
        else:
            out_nonzero_path = out_path + '_nonzero.png'

    plt.figure(figsize=(8, 5))
    for k in sorted(attributes_dict.keys()):
        selected = attributes_dict[k]
        nz = []
        for lam in lambdas:
            _, avg_nz, _ = compute_average_metrics(modified_rewards, selected, lam)
            nz.append(avg_nz)
        plt.plot(lambdas, nz, marker='s', label=f"k={k}")

    plt.xlabel("l1_lambda")
    plt.ylabel("Average non-zero weights")
    plt.title("Average non-zero weights vs l1_lambda across attribute counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_nonzero_path, dpi=150)
    print(f"Saved non-zero count sweep plot to {out_nonzero_path}")


def plot_by_k_with_formula_lambda(modified_rewards, attributes_dict, out_path, out_nonzero_path=None):
    """
    For each attribute count k, set l1_lambda = sqrt(log(k)/k) and compute average
    accuracy and average number of non-zero weights across users. Plot metrics vs k.
    """
    ks = sorted(attributes_dict.keys())
    accs = []
    nzs = []
    lambdas_used = []
    for k in ks:
        selected = attributes_dict[k]
        lam = math.sqrt(max(math.log(k), 0.0) / float(k))
        lambdas_used.append(lam)
        avg_acc, avg_nz, n_users = compute_average_metrics(modified_rewards, selected, lam)
        accs.append(avg_acc)
        nzs.append(avg_nz)

    # Accuracy vs k
    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, marker='o')
    plt.xlabel("k (number of attributes)")
    plt.ylabel("Average test accuracy")
    plt.title("Average accuracy vs k with l1_lambda = sqrt(log(k)/k)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved k-curve accuracy plot to {out_path}")

    # Non-zero vs k
    if out_nonzero_path is None:
        if out_path.lower().endswith('.png'):
            out_nonzero_path = out_path[:-4] + '_nonzero.png'
        else:
            out_nonzero_path = out_path + '_nonzero.png'

    plt.figure(figsize=(8, 5))
    plt.plot(ks, nzs, marker='s')
    plt.xlabel("k (number of attributes)")
    plt.ylabel("Average non-zero weights")
    plt.title("Average non-zero weights vs k with l1_lambda = sqrt(log(k)/k)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_nonzero_path, dpi=150)
    print(f"Saved k-curve non-zero plot to {out_nonzero_path}")

    # Combined figure for the single-schedule variant
    if out_path.lower().endswith('.png'):
        out_combined_path = out_path[:-4] + '_combined.png'
    else:
        out_combined_path = out_path + '_combined.png'

    # Combined figure disabled by preference; keep separate plots only


def _estimate_n_samples(modified_rewards):
    """Estimate dataset size n from rewards (average over users)."""
    ns = []
    for user_id, r in modified_rewards.items():
        try:
            ns.append(int(r['attr_scores_chosen'].shape[0]))
        except Exception:
            continue
    return int(np.mean(ns)) if ns else 0


def _lambda_for_schedule(schedule: str, k: int, c: float, kmax: int, n: int, sigma: float | None = None) -> float:
    """Compute lambda for a given schedule name."""
    if k <= 0:
        return 0.0
    s = schedule.lower()
    if s in ("sqrtlog_over_k", "sqrt_log_over_k", "sqrtlogk_over_k"):
        return c * math.sqrt(max(math.log(1 + k), 0.0) / float(k))
    if s in ("linear", "k_over_kmax"):
        denom = max(kmax, 1)
        return c * (float(k) / float(denom))
    if s in ("sqrtlog", "sqrt_log"):
        return c * math.sqrt(max(math.log(1 + k), 0.0))
    if s in ("k_over_n", "linear_over_n"):
        denom = max(n, 1)
        return c * (float(k) / float(denom))
    if s in ("log_over_n", "logk_over_n"):
        denom = max(n, 1)
        return c * (math.log(1 + k) / float(denom))
    if s in ("sqrtlog_over_n", "sqrt_log_over_n"):
        denom = max(n, 1)
        return c * math.sqrt(max(math.log(1 + k), 0.0) / float(denom))
    if s in ("sigma_sqrtlog_over_n", "sigma_sqrt_log_over_n"):
        denom = max(n, 1)
        sig = sigma if sigma is not None else 1.0
        return c * sig * math.sqrt(max(math.log(1 + k), 0.0) / float(denom))
    if s in ("sigma_log_over_n",):
        denom = max(n, 1)
        sig = sigma if sigma is not None else 1.0
        return c * sig * (math.log(1 + k) / float(denom))
    if s in ("sigma_over_sqrtn", "sigma_over_sqrt_n"):
        denom = max(n, 1)
        sig = sigma if sigma is not None else 1.0
        return c * sig / math.sqrt(float(denom))
    # Fallback to sqrtlog_over_k
    return c * math.sqrt(max(math.log(1 + k), 0.0) / float(k))


def _estimate_sigma_global(modified_rewards, selected_attributes):
    """Estimate a global noise scale sigma from X matrices over the first available user."""
    # pick first user deterministically
    for test_user_id in sorted(modified_rewards.keys()):
        try:
            # Build chosen/rejected log probs using same logic as in approximate_local
            eps = 1e-12
            r = modified_rewards[test_user_id]
            chosen_attr_probs = r['attr_scores_chosen'] / r['attr_counts_chosen'].clamp(min=eps)
            chosen_base_probs = r['base_scores_chosen'].unsqueeze(1) / r['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
            chosen_log_probs = chosen_attr_probs - chosen_base_probs
            rejected_attr_probs = r['attr_scores_rejected'] / r['attr_counts_rejected'].clamp(min=eps)
            rejected_base_probs = r['base_scores_rejected'].unsqueeze(1) / r['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
            rejected_log_probs = rejected_attr_probs - rejected_base_probs
            if selected_attributes is not None:
                idx = torch.tensor(selected_attributes, dtype=torch.long)
                chosen_log_probs = chosen_log_probs[:, idx]
                rejected_log_probs = rejected_log_probs[:, idx]
            X = (chosen_log_probs - rejected_log_probs).detach().cpu().numpy()
            return float(X.std()) if X.size > 0 else 1.0
        except Exception:
            continue
    return 1.0


def plot_by_k_multi(modified_rewards, attributes_dict, schedules, c, kmax, n, lambdas_for_max, out_path, out_nonzero_path=None):
    """
    Plot accuracy and non-zero vs k for multiple schedules. If a schedule is 'max',
    pick the best accuracy across lambdas_for_max for each k.
    """
    ks = sorted(attributes_dict.keys())
    lines_acc = {}
    lines_nz = {}

    for sched in schedules:
        accs = []
        nzs = []
        if sched.lower() == 'max':
            # Hyperparameter search per k
            for k in ks:
                selected = attributes_dict[k]
                best_acc = -1.0
                best_nz = 0.0
                for lam in lambdas_for_max:
                    avg_acc, avg_nz, _ = compute_average_metrics(modified_rewards, selected, lam)
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_nz = avg_nz
                accs.append(best_acc if best_acc >= 0.0 else 0.0)
                nzs.append(best_nz)
        else:
            for k in ks:
                selected = attributes_dict[k]
                sigma = _estimate_sigma_global(modified_rewards, selected) if 'sigma' in sched.lower() else None
                lam = _lambda_for_schedule(sched, k, c, kmax, n, sigma)
                avg_acc, avg_nz, _ = compute_average_metrics(modified_rewards, selected, lam)
                accs.append(avg_acc)
                nzs.append(avg_nz)
        lines_acc[sched] = accs
        lines_nz[sched] = nzs

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    for sched, accs in lines_acc.items():
        plt.plot(ks, accs, marker='o', label=sched)
    plt.xlabel("k (number of attributes)")
    plt.ylabel("Average test accuracy")
    plt.title("Accuracy vs k for multiple lambda schedules")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved multi-schedule k-curve accuracy plot to {out_path}")

    # Non-zero plot
    if out_nonzero_path is None:
        if out_path.lower().endswith('.png'):
            out_nonzero_path = out_path[:-4] + '_nonzero.png'
        else:
            out_nonzero_path = out_path + '_nonzero.png'

    plt.figure(figsize=(8, 5))
    for sched, nzs in lines_nz.items():
        plt.plot(ks, nzs, marker='s', label=sched)
    plt.xlabel("k (number of attributes)")
    plt.ylabel("Average non-zero weights")
    plt.title("Non-zero weights vs k for multiple lambda schedules")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_nonzero_path, dpi=150)
    print(f"Saved multi-schedule k-curve non-zero plot to {out_nonzero_path}")

    # Combined figure disabled by preference; keep separate plots only


def main(selected_attributes=None, l1_lambda=0.01, seed=0):
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

    # Load and modify rewards
    modified_rewards = load_and_modify_rewards()
    
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
                selected_attributes=selected_attributes
            )
            
            # Evaluate accuracy
            accuracy = evaluate_accuracy_local(
                modified_rewards, 
                test_user_id, 
                p, 
                train_indices, 
                test_indices,
                selected_attributes=selected_attrs
            )
            
            all_accuracies.append(accuracy)
            all_nonzero_weights.append(np.count_nonzero(p))
            
            print(f"\nResults for {test_user_id}:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Non-zero weights: {np.count_nonzero(p)}/{len(p)}")
            # Print learned weights vector
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
        if selected_attributes is not None:
            print(f"Selected attributes: {selected_attributes}")
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
    30: [30, 140, 270, 260, 117, 153, 56, 393, 390, 380, 87, 187, 331, 298, 49, 207, 363, 198, 172, 110, 52, 59, 120, 74, 225, 293, 373, 83, 114, 370],
    60: [330, 238, 130, 81, 187, 332, 146, 185, 49, 341, 397, 95, 225, 62, 393, 56, 2, 260, 293, 249, 74, 128, 112, 178, 55, 371, 83, 256, 362, 136, 152, 15, 13, 311, 193, 69, 285, 117, 207, 377, 27, 337, 101, 110, 176, 119, 275, 87, 91, 75, 70, 77, 265, 309, 198, 168, 133, 53, 390, 135],
    120: [8, 387, 56, 196, 133, 2, 310, 126, 249, 73, 338, 95, 127, 218, 155, 232, 380, 9, 168, 190, 86, 393, 387, 66, 87, 122, 295, 346, 245, 396, 257, 83, 145, 243, 89, 170, 46, 343, 252, 120, 16, 153, 103, 363, 390, 78, 147, 119, 162, 90, 112, 260, 91, 399, 194, 273, 178, 54, 303, 327, 305, 161, 198, 13, 332, 227, 152, 48, 141, 375, 39, 121, 360, 114, 283, 100, 308, 397, 146, 304, 125, 49, 254, 262, 169, 369, 106, 58, 59, 223, 207, 176, 80, 330, 275, 158, 27, 302, 34, 3, 182, 5, 307, 188, 104, 225, 225, 355, 270, 319, 199, 29, 117, 364, 128, 134, 344, 144, 378, 277],
    180: [104, 29, 13, 38, 15, 2, 292, 136, 187, 284, 374, 249, 302, 340, 199, 178, 165, 219, 39, 90, 369, 287, 172, 34, 308, 281, 114, 265, 72, 77, 389, 144, 114, 176, 342, 349, 390, 262, 360, 91, 27, 62, 244, 142, 384, 123, 152, 341, 113, 296, 330, 162, 64, 224, 98, 278, 139, 49, 256, 147, 392, 236, 121, 120, 134, 300, 257, 186, 331, 377, 225, 288, 343, 285, 67, 20, 332, 225, 222, 364, 55, 295, 273, 338, 56, 399, 252, 115, 158, 130, 242, 275, 76, 188, 206, 92, 85, 175, 16, 48, 101, 168, 89, 131, 397, 198, 70, 140, 135, 35, 260, 316, 118, 386, 74, 27, 96, 232, 23, 254, 169, 63, 325, 57, 214, 276, 155, 52, 352, 339, 183, 60, 107, 99, 309, 243, 46, 148, 218, 350, 37, 91, 270, 393, 370, 45, 399, 10, 66, 73, 190, 59, 371, 355, 78, 128, 103, 266, 146, 213, 19, 387, 228, 333, 261, 145, 12, 327, 251, 95, 29, 307, 305, 83, 346, 361, 358, 319, 17, 30],
    240: [299, 122, 168, 165, 68, 385, 397, 17, 182, 389, 330, 316, 265, 99, 237, 91, 266, 222, 285, 193, 399, 112, 97, 179, 140, 20, 361, 30, 216, 91, 17, 59, 181, 249, 189, 139, 138, 152, 56, 291, 29, 364, 225, 127, 71, 300, 109, 302, 346, 272, 243, 61, 334, 309, 49, 219, 133, 319, 225, 146, 123, 103, 66, 2, 270, 390, 307, 74, 37, 75, 260, 151, 304, 144, 387, 82, 236, 106, 350, 114, 39, 207, 264, 81, 43, 72, 120, 203, 15, 257, 160, 356, 145, 262, 124, 167, 386, 58, 94, 136, 346, 396, 134, 27, 395, 208, 252, 155, 231, 355, 263, 344, 26, 95, 254, 351, 55, 35, 69, 159, 31, 174, 188, 322, 27, 274, 76, 190, 163, 162, 119, 352, 276, 238, 327, 298, 313, 121, 377, 343, 13, 172, 73, 64, 13, 170, 372, 244, 342, 19, 118, 44, 261, 57, 234, 301, 89, 310, 9, 132, 338, 87, 359, 198, 275, 176, 16, 27, 247, 135, 72, 306, 62, 42, 196, 223, 161, 371, 369, 47, 105, 9, 290, 363, 104, 325, 311, 374, 232, 102, 77, 23, 86, 153, 193, 278, 349, 270, 375, 48, 308, 146, 8, 295, 111, 256, 185, 336, 303, 178, 228, 162, 169, 6, 130, 187, 100, 208, 90, 245, 213, 360, 70, 78, 147, 294, 391, 212, 394, 380, 259, 11, 215, 253, 92, 201, 227, 357, 153, 285]
}

if __name__ == "__main__":
    import argparse
    import random

    # Default selected attributes remain as before; set to None to use all attributes
    selected_attributes = []

    parser = argparse.ArgumentParser(description="Evaluate selection accuracy, print learned weights, and generate plots.")
    parser.add_argument("--l1_lambda", type=float, default=0.01, help="L1 regularization strength (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--plot_sweep", action="store_true", help="Plot average accuracy vs lambda for multiple attribute counts.")
    parser.add_argument("--plot_by_k", action="store_true", help="Plot metrics vs k using l1_lambda = sqrt(log(k)/k) per k.")
    parser.add_argument("--plot_by_k_multi", action="store_true", help="Plot metrics vs k for multiple lambda schedules, including 'max'.")
    parser.add_argument("--lambdas", type=str, default="", help="Comma-separated list of lambdas to sweep (e.g., 0.001,0.003,0.01). If provided, overrides min/max/num settings.")
    parser.add_argument("--lambda_min", type=float, default=1e-4, help="Minimum lambda for generated grid (default: 1e-4)")
    parser.add_argument("--lambda_max", type=float, default=1.0, help="Maximum lambda for generated grid (default: 1.0)")
    parser.add_argument("--num_lambdas", type=int, default=41, help="Number of lambdas to generate (default: 41)")
    parser.add_argument("--lambda_scale", choices=["log", "lin"], default="log", help="Spacing of generated lambdas (default: log)")
    parser.add_argument("--out", type=str, default="lambda_sweep.png", help="Output path for the accuracy sweep plot.")
    parser.add_argument("--out_nonzero", type=str, default="", help="Output path for the non-zero weights sweep plot (default: derives from --out).")
    parser.add_argument("--schedules", type=str, default="sqrtlog_over_k,linear,max", help="Comma-separated schedules for --plot_by_k_multi (e.g., sqrtlog_over_k,linear,k_over_n,log_over_n,max)")
    parser.add_argument("--c", type=float, default=1.0, help="Scale factor c used in schedule formulas (default: 1.0)")
    parser.add_argument("--kmax", type=int, default=0, help="k_max used by 'linear' schedule; defaults to max key in attributes.")
    parser.add_argument("--n", type=int, default=0, help="n used by *_over_n schedules; defaults to dataset size estimate.")
    args = parser.parse_args()

    if args.plot_sweep:
        # Build lambda grid
        if args.lambdas.strip():
            lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
        else:
            # Generate a dense grid
            if args.lambda_scale == "log":
                lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), num=args.num_lambdas).tolist()
            else:
                lambdas = np.linspace(args.lambda_min, args.lambda_max, num=args.num_lambdas).tolist()

        main(selected_attributes=None, l1_lambda=args.l1_lambda, seed=args.seed)  # Optional single run printout
        # Recreate rewards under the same seed to ensure consistency
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        modified_rewards = load_and_modify_rewards()
        out_nz = args.out_nonzero if args.out_nonzero.strip() else None
        sweep_lambdas_and_plot(modified_rewards, attributes, lambdas, args.out, out_nz)
    elif args.plot_by_k:
        # Create rewards once under the provided seed
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        modified_rewards = load_and_modify_rewards()
        out_nz = args.out_nonzero if args.out_nonzero.strip() else None
        plot_by_k_with_formula_lambda(modified_rewards, attributes, args.out, out_nz)
    elif args.plot_by_k_multi:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        modified_rewards = load_and_modify_rewards()
        out_nz = args.out_nonzero if args.out_nonzero.strip() else None
        # schedules list
        schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]
        # resolve kmax and n
        kmax = args.kmax if args.kmax > 0 else max(attributes.keys())
        n_est = args.n if args.n > 0 else _estimate_n_samples(modified_rewards)
        # lambdas grid used for 'max'
        if args.lambdas.strip():
            lambdas_for_max = [float(x) for x in args.lambdas.split(',') if x.strip()]
        else:
            if args.lambda_scale == "log":
                lambdas_for_max = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), num=args.num_lambdas).tolist()
            else:
                lambdas_for_max = np.linspace(args.lambda_min, args.lambda_max, num=args.num_lambdas).tolist()
        plot_by_k_multi(modified_rewards, attributes, schedules, args.c, kmax, n_est, lambdas_for_max, args.out, out_nz)
    else:
        # Default behavior: run comprehensive multi-schedule k-curves and also the accuracy/nonzero-by-k with sqrtlog_over_k
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        modified_rewards = load_and_modify_rewards()
        out_nz = args.out_nonzero if args.out_nonzero.strip() else None

        # Defaults
        schedules = [
            "sqrtlog_over_k",
            "linear",
            "k_over_n",
            "log_over_n",
            "sqrtlog_over_n",
            "sigma_sqrtlog_over_n",
            "sigma_log_over_n",
            "sigma_over_sqrtn",
            "max",
        ]
        kmax = max(attributes.keys())
        n_est = _estimate_n_samples(modified_rewards)
        # grid for 'max'
        lambdas_for_max = np.logspace(np.log10(1e-5), np.log10(1.0), num=101).tolist()
        plot_by_k_multi(modified_rewards, attributes, schedules, 1.0, kmax, n_est, lambdas_for_max, args.out, out_nz)

        # Additionally, produce the single-formula sqrtlog_over_k curves for backward-compat naming
        plot_by_k_with_formula_lambda(modified_rewards, attributes, args.out.replace('.png', '_sqrtlogk_over_k.png'), None)
