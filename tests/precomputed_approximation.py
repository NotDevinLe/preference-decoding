import torch
import numpy as np
import random

def l1_solve(d_mean, l1_lambda, std=None):
    """
    Closed-form solution to: maximize d^T p - lambda * ||p||_1  s.t. ||p||_2 <= 1
    """
    d = np.asarray(d_mean, dtype=float)
    z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
    norm = np.linalg.norm(z, ord=2)
    if norm == 0.0:
        return np.zeros_like(d)
    if std is None:
        return z / norm
    else:
        return z / (norm * std)


def compute_normalized_log_probs(rewards):
    """
    Compute normalized log probabilities following the derived formulation.
    """
    eps = 1e-12
    attr_log_probs = (rewards['attr_scores_chosen'] / 
                      rewards['attr_counts_chosen'].clamp(min=eps))
    
    base_log_probs = (rewards['base_scores_chosen'].unsqueeze(1) / 
                      rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps))
    
    return attr_log_probs - base_log_probs


def compute_drift_weights(selected_attributes, test_user_id, num_users=120, train_ratio=0.8, 
                         l1_lambda=0.01, base_dir="rewards_persona_testing_small"):
    """
    Compute drift approximation weights with train/test split using L1 regularization.
    Creates preference pairs: test_user vs each other user for each prompt.
    
    Args:
        selected_attributes: List or tensor of attribute indices to use (e.g., [0, 2, 5, 7])
                            If None, uses all attributes
        test_user_id: ID of test user
        num_users: Total number of users
        train_ratio: Train/test split ratio
        l1_lambda: L1 regularization parameter
        base_dir: Directory containing reward files
    """
    # print(f"\n{'='*60}")
    # print(f"Processing user {test_user_id}")
    # print(f"{'='*60}")
    
    # Load test user (chosen) rewards
    test_rewards_path = f"{base_dir}/user{test_user_id}.pt"
    test_rewards = torch.load(test_rewards_path)
    
    # Compute chosen log probs
    chosen_log_probs = compute_normalized_log_probs(test_rewards)
    n_prompts, k_attributes_full = chosen_log_probs.shape
    
    # Filter attributes if specified
    if selected_attributes is not None:
        if isinstance(selected_attributes, list):
            selected_attributes = torch.tensor(selected_attributes, dtype=torch.long)
        chosen_log_probs = chosen_log_probs[:, selected_attributes]
        k_attributes = len(selected_attributes)
        # print(f"Using {k_attributes} selected attributes out of {k_attributes_full}: {selected_attributes.tolist()}")
    else:
        k_attributes = k_attributes_full
        # print(f"Using all {k_attributes} attributes")
    
    # print(f"Test user {test_user_id} has {n_prompts} prompts and {k_attributes} attributes")
    
    # Collect all rejected users' data
    rejected_log_probs_all_users = []
    rejected_user_ids = []
    
    for train_user_id in range(num_users):
        if train_user_id == test_user_id:
            continue
            
        train_rewards_path = f"{base_dir}/user{train_user_id}.pt"
        try:
            train_rewards = torch.load(train_rewards_path)
            rejected_log_probs = compute_normalized_log_probs(train_rewards)
            
            # Filter attributes if specified
            if selected_attributes is not None:
                rejected_log_probs = rejected_log_probs[:, selected_attributes]
            
            # Verify shape matches
            if rejected_log_probs.shape[0] != n_prompts:
                print(f"  WARNING: User {train_user_id} has {rejected_log_probs.shape[0]} prompts (expected {n_prompts}), skipping")
                continue
                
            rejected_log_probs_all_users.append(rejected_log_probs)
            rejected_user_ids.append(train_user_id)
        except Exception as e:
            print(f"  WARNING: Could not load user {train_user_id}: {e}")
            continue
    
    # Stack: (num_other_users, n_prompts, k_attributes)
    rejected_log_probs_all = torch.stack(rejected_log_probs_all_users)
    num_other_users = rejected_log_probs_all.shape[0]
    
    # print(f"Loaded {num_other_users} other users")
    # print(f"rejected_log_probs_all shape: {rejected_log_probs_all.shape}")
    # print(f"chosen_log_probs shape: {chosen_log_probs.shape}")
    
    # Split prompts into train/test
    n_train_prompts = int(n_prompts * train_ratio)
    n_test_prompts = n_prompts - n_train_prompts
    
    train_prompt_indices = list(range(0, n_train_prompts))
    test_prompt_indices = list(range(n_train_prompts, n_prompts))
    
    n_train_pairs = n_train_prompts * num_other_users
    n_test_pairs = n_test_prompts * num_other_users
    
    # print(f"\nTrain: {n_train_prompts} prompts × {num_other_users} users = {n_train_pairs} pairs")
    # print(f"Test:  {n_test_prompts} prompts × {num_other_users} users = {n_test_pairs} pairs")
    
    # ============ TRAINING PHASE ============
    # Build X matrix: each row is a preference pair
    X_train = torch.zeros(n_train_pairs, k_attributes, dtype=torch.float32)
    
    # print(f"\nBuilding X_train matrix with shape {X_train.shape}...")
    pair_idx = 0
    for prompt_idx in train_prompt_indices:
        for user_idx in range(num_other_users):
            # Drift for this preference pair
            chosen_drift = chosen_log_probs[prompt_idx]
            rejected_drift = rejected_log_probs_all[user_idx, prompt_idx, :]
            
            X_train[pair_idx] = chosen_drift - rejected_drift
            pair_idx += 1
    
    # print(f"Filled {pair_idx} rows in X_train (expected {n_train_pairs})")
    assert pair_idx == n_train_pairs, f"Mismatch! pair_idx={pair_idx}, n_train_pairs={n_train_pairs}"
    
    # Compute column-wise statistics for standardization
    col_std = X_train.std(dim=0).clamp(min=1e-8)
    
    # Standardize and compute mean drift direction
    X_standardized = X_train / col_std
    d = X_standardized.mean(dim=0).detach().cpu().numpy()
    
    # Solve for p vector with L1 regularization
    p = l1_solve(d, l1_lambda, std=col_std.detach().cpu().numpy())
    p_tensor = torch.tensor(p, dtype=torch.float32)
    
    # print(f"\nComputed p vector with {np.count_nonzero(p)} non-zero weights")
    
    # ============ EVALUATION PHASE ============
    
    # Evaluate on training pairs
    train_drift_scores = X_train @ p_tensor
    train_correct = (train_drift_scores > 0).sum().item()
    train_accuracy = train_correct / n_train_pairs
    train_mean_score = train_drift_scores.mean().item()
    
    # print(f"\nTrain Results:")
    # print(f"  Accuracy: {train_accuracy:.4f} ({train_correct}/{n_train_pairs})")
    # print(f"  Mean score: {train_mean_score:.4f}")
    
    # Evaluate on test pairs
    X_test = torch.zeros(n_test_pairs, k_attributes, dtype=torch.float32)
    
    # print(f"\nBuilding X_test matrix with shape {X_test.shape}...")
    pair_idx = 0
    for prompt_idx in test_prompt_indices:
        for user_idx in range(num_other_users):
            chosen_drift = chosen_log_probs[prompt_idx]
            rejected_drift = rejected_log_probs_all[user_idx, prompt_idx, :]
            
            X_test[pair_idx] = chosen_drift - rejected_drift
            pair_idx += 1
    
    # print(f"Filled {pair_idx} rows in X_test (expected {n_test_pairs})")
    assert pair_idx == n_test_pairs, f"Mismatch! pair_idx={pair_idx}, n_test_pairs={n_test_pairs}"
    
    test_drift_scores = X_test @ p_tensor
    test_correct = (test_drift_scores > 0).sum().item()
    test_accuracy = test_correct / n_test_pairs
    test_mean_score = test_drift_scores.mean().item()
    
    # print(f"\nTest Results:")
    # print(f"  Accuracy: {test_accuracy:.4f} ({test_correct}/{n_test_pairs})")
    # print(f"  Mean score: {test_mean_score:.4f}")
    
    # Get top attributes by absolute weight
    top_k = min(5, k_attributes)
    top_indices = np.argsort(np.abs(p))[-top_k:][::-1]

    return test_accuracy, train_accuracy


if __name__ == "__main__":
    
    selected_attributes = [331, 333, 311, 91, 157]
    avg_test, avg_train = 0, 0
    for i in range(120):
        test_acc, train_acc = compute_drift_weights(
            selected_attributes=selected_attributes, 
            test_user_id=i, 
            num_users=120, 
            train_ratio=0.8, 
            l1_lambda=0.1
        )
        avg_test += test_acc
        avg_train += train_acc
    
    print(f"\nAverage Test Accuracy: {avg_test / 120:.4f}")
    print(f"Average Train Accuracy: {avg_train / 120:.4f}")