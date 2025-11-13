import os
import glob
import random as _random
import argparse
import numpy as np
import torch
import json
from collections import defaultdict, Counter

def compute_normalized_log_probs(rewards):
    eps = 1e-12    
    chosen_attr = rewards['attr_scores_chosen'] / rewards['attr_counts_chosen'].clamp(min=eps)
    chosen_base = rewards['base_scores_chosen'].unsqueeze(1) / rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
    rejected_attr = rewards['attr_scores_rejected'] / rewards['attr_counts_rejected'].clamp(min=eps)
    rejected_base = rewards['base_scores_rejected'].unsqueeze(1) / rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
    
    chosen_log = chosen_attr - chosen_base
    rejected_log = rejected_attr - rejected_base
    return chosen_log, rejected_log


def load_and_modify_rewards(rewards_dir="rewards_persona_testing_testingset", num_users=None):
    pattern = "user*.pt"
    reward_files = glob.glob(os.path.join(rewards_dir, pattern))
    reward_files.sort()
    
    if len(reward_files) == 0:
        print(f"No reward files found in {rewards_dir}")
        return {}
    
    if num_users is not None:
        reward_files = reward_files[:num_users]
    print(f"Processing first {len(reward_files)} reward files from {rewards_dir}\n")
    
    all_rewards = {}
    user_ids = []
    
    for reward_file in reward_files:
        user_id = os.path.basename(reward_file).replace('.pt', '')
        user_ids.append(user_id)
        rewards = torch.load(reward_file, map_location="cpu")
        all_rewards[user_id] = rewards
        print(f"Loaded {user_id}: {rewards['attr_scores_chosen'].shape[0]} samples")
    
    print(f"\nProcessing {len(user_ids)} users...")
    
    modified_rewards = {}
    
    for user_id in user_ids:
        print(f"\nProcessing {user_id}...")
        
        user_rewards = all_rewards[user_id]
        n_samples = user_rewards['attr_scores_chosen'].shape[0]
        
        other_users = [uid for uid in user_ids if uid != user_id]
        
        modified = {}
        for key in user_rewards.keys():
            if isinstance(user_rewards[key], dict):
                modified[key] = user_rewards[key].copy()
            elif hasattr(user_rewards[key], 'clone'):
                modified[key] = user_rewards[key].clone()
            else:
                modified[key] = user_rewards[key]
        
        for sample_idx in range(n_samples):
            sampled_user = _random.choice(other_users)
            sampled_rewards = all_rewards[sampled_user]
            
            if sample_idx < sampled_rewards['attr_scores_chosen'].shape[0]:
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


def accuracy_from_delta(delta_phi: torch.Tensor, p_t: torch.Tensor) -> float:
    scores = delta_phi @ p_t
    correct = (scores > 0).sum().item()
    return correct / delta_phi.shape[0]


def evaluate_single_user(
    rewards,
    user_id,
    p: np.ndarray,
    swap_chosen_rejected: bool = False,
    normalize_features: bool = False,
    gaussian_noise_std: float = 0.0,
    seed: int | None = None,
    verbose: bool = True,
):
    if seed is not None:
        torch.manual_seed(seed)
    
    r = rewards[user_id]
    chosen_log, rejected_log = compute_normalized_log_probs(r)

    delta = chosen_log - rejected_log
    if swap_chosen_rejected:
        delta = -delta

    if normalize_features:
        delta = (delta - delta.mean(0, keepdim=True)) / (delta.std(0, keepdim=True) + 1e-8)

    if gaussian_noise_std > 0:
        noise = torch.randn_like(delta) * gaussian_noise_std
        delta = delta + noise

    if verbose:
        print(f"\n=== {user_id} ===")
        print(f"Num pairs: {delta.shape[0]}, num attrs: {delta.shape[1]}")

    p_t = torch.tensor(p, dtype=torch.float32)
    delta = delta.to(dtype=torch.float32)
    acc = accuracy_from_delta(delta, p_t)

    if verbose:
        print(f"{user_id}: accuracy={acc:.4f}")
    
    return acc


def load_p_jsonl(path: str) -> dict:
    """
    Load a JSONL file mapping user -> p vector.
    Flattens p with ravel() so nested lists like [[a,b]] become [a,b].
    """
    p_map = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = obj["user"]
            p_arr = np.array(obj["p"], dtype=np.float32).ravel()  # <-- flatten
            p_map[user] = p_arr
    return p_map


def run(args):
    _random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    rewards = load_and_modify_rewards(args.rewards_dir, num_users=None)
    if not rewards:
        print(f"No reward files found in {args.rewards_dir}")
        return 1

    p_map_all = load_p_jsonl(args.p_jsonl)
    print(f"Loaded p for {len(p_map_all)} users from {args.p_jsonl}")

    # Debug: show distribution of p lengths in this JSONL
    length_counts = Counter(len(p) for p in p_map_all.values())
    print("\nDistribution of len(p) in JSONL:")
    for length, count in sorted(length_counts.items()):
        print(f"  k={length}: {count} users")

    k = args.k
    print(f"\nFiltering to p vectors with length k = {k}")

    p_map = {u: p for u, p in p_map_all.items() if p.shape[0] == k}
    print(f"Number of users with p dim = {k}: {len(p_map)}")

    if not p_map:
        print("No p vectors found with the requested k. Exiting.")
        return 1

    all_acc = []
    used_users = 0

    for user_id in rewards.keys():
        if user_id not in p_map:
            continue

        p = p_map[user_id]

        num_attrs = rewards[user_id]["attr_scores_chosen"].shape[1]
        if num_attrs != k:
            print(f"Skipping {user_id}: p dim={k} but reward attr dim={num_attrs}")
            continue

        acc = evaluate_single_user(
            rewards,
            user_id,
            p,
            swap_chosen_rejected=False,
            normalize_features=False,
            gaussian_noise_std=0.0,
            seed=None,
            verbose=True,
        )
        all_acc.append(acc)
        used_users += 1

    if all_acc:
        mean_acc = float(np.mean(all_acc))
        std_acc = float(np.std(all_acc))
        print(f"\nAvg accuracy for k={k} across {used_users} users: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print(f"No users evaluated for k={k} (no matching p + reward dims).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate per-user p (from JSONL) on modified rewards, filtering by k=len(p)."
    )
    parser.add_argument("--rewards_dir", type=str, default="rewards_persona_testing_testingset_10")
    parser.add_argument(
        "--p_jsonl",
        type=str,
        required=True,
        help='Path to JSONL file with lines like: {"user": "user0", "p": [...]}',
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Only evaluate users whose p vector has length k and whose reward attr dim is k.",
    )
    parser.add_argument("--use_cuda", action="store_true")  # unused; left for compatibility

    args = parser.parse_args()
    run(args)