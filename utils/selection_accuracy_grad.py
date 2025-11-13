import os
import glob
import random as _random
import argparse
import numpy as np
import torch
import json

results = []

def compute_normalized_log_probs(rewards, selected_attributes):
    eps = 1e-12    
    chosen_attr = rewards['attr_scores_chosen'][:,selected_attributes] / rewards['attr_counts_chosen'][:, selected_attributes].clamp(min=eps)
    chosen_base = rewards['base_scores_chosen'].unsqueeze(1) / rewards['base_counts_chosen'].unsqueeze(1).clamp(min=eps)
    rejected_attr = rewards['attr_scores_rejected'][:, selected_attributes] / rewards['attr_counts_rejected'][:, selected_attributes].clamp(min=eps)
    rejected_base = rewards['base_scores_rejected'].unsqueeze(1) / rewards['base_counts_rejected'].unsqueeze(1).clamp(min=eps)
    
    chosen_log = chosen_attr - chosen_base
    rejected_log = rejected_attr - rejected_base
    return chosen_log, rejected_log


def load_rewards_dir(rewards_dir: str, num_users: int | None = None):
    pattern = os.path.join(rewards_dir, "user*.pt")
    files = sorted(glob.glob(pattern))
    if num_users is not None:
        files = files[: num_users]
    rewards_by_user = {}
    for f in files:
        uid = os.path.basename(f).replace(".pt", "")
        rewards_by_user[uid] = torch.load(f, map_location="cpu")
    return rewards_by_user

def bt_loss_with_l1(delta_phi: torch.Tensor, p: torch.Tensor, l1_lambda: float, beta: float = 1.0):
    """Bradley-Terry negative log-likelihood + L1."""
    logits = beta * (delta_phi @ p)
    nll = -torch.nn.functional.logsigmoid(logits).mean()
    l1_penalty = l1_lambda * torch.norm(p, 1)
    return nll + l1_penalty, nll, l1_penalty


def accuracy_from_delta(delta_phi: torch.Tensor, p: torch.Tensor) -> float:
    scores = delta_phi @ p
    correct = (scores > 0).sum().item()
    return correct / (delta_phi.shape[0] + 1e-8)

def optimize_p(delta_phi: torch.Tensor,
               epochs: int,
               lr: float,
               l1_lambda: float,
               device: torch.device,
               beta: float = 1.0,
               verbose: bool = True):
    """Train p on delta_phi using BT + L1."""
    delta_phi = delta_phi.to(device)
    n, k = delta_phi.shape
    if verbose:
        print(delta_phi.shape)
    p = torch.randn(k, device=device, requires_grad=True)
    with torch.no_grad():
        norm = p.norm()
        if norm > 0:
            p /= norm

    optimizer = torch.optim.Adam([p], lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, nll, l1_penalty = bt_loss_with_l1(delta_phi, p, l1_lambda, beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = p.norm()
            train_acc = accuracy_from_delta(delta_phi, p)
        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "nll": float(nll.item()),
            "l1": float(l1_penalty.item()),
            "train_acc": float(train_acc),
        })
    return p.detach().cpu().numpy().copy(), history


def evaluate_accuracy(delta_phi_test: torch.Tensor, p: np.ndarray) -> float:
    p_t = torch.tensor(p, dtype=torch.float32)
    scores = delta_phi_test @ p_t
    correct = (scores > 0).sum().item()
    return correct / delta_phi_test.shape[0]


def evaluate_single_user(rewards, user_id, selected_attributes, train_ratio, epochs, lr, l1_lambda, 
                         device, beta, seed=None, verbose=True, leave_one_out=False):
    """Evaluate a single user and return test accuracy.
    
    Args:
        leave_one_out: If True, perform leave-one-out cross-validation where each sample
                       is used as test set once, training on all other samples.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    r = rewards[user_id]
    chosen_log, rejected_log = compute_normalized_log_probs(r, selected_attributes)

    n = chosen_log.shape[0]
    
    if leave_one_out:
        if verbose:
            print(f"\n=== {user_id} (Leave-One-Out) ===")
            print(f"Performing leave-one-out evaluation on {n} samples...")
        
        delta = chosen_log - rejected_log
        all_accuracies = []
        all_ps = []
        
        for test_idx in range(n):
            if verbose:
                print(f"\n  Fold {test_idx + 1}/{n} (testing on sample {test_idx})...")

            train_indices = [i for i in range(n) if i != test_idx]
            delta_train = delta[train_indices]
            delta_test = delta[test_idx:test_idx+1]
            
            p, history = optimize_p(delta_train, epochs, lr, l1_lambda, device, beta, verbose=verbose)
            
            if verbose:
                print(f"    p: {p}")
                for h in history:
                    print(f"    epoch {h['epoch']:03d} loss={h['loss']:.4f} "
                          f"nll={h['nll']:.4f} l1={h['l1']:.4f} train_acc={h['train_acc']:.4f}")
            
            acc = evaluate_accuracy(delta_test, p)
            all_accuracies.append(acc)
            all_ps.append(p)
            
            if verbose:
                print(f"    Test accuracy (sample {test_idx}): {acc:.4f}")
        
        avg_acc = np.mean(all_accuracies)
        avg_p = np.mean(all_ps, axis=0)
        
        if verbose:
            print(f"{user_id}: leave_one_out_acc={avg_acc:.4f} (n_samples={n})")
            print(f"  Accuracy range: [{min(all_accuracies):.4f}, {max(all_accuracies):.4f}]")
        
        return avg_acc, avg_p
    else:
        perm = torch.randperm(n)
        chosen_log = chosen_log[perm]
        rejected_log = rejected_log[perm]

        n_train = int(n * train_ratio)
        train_idx, test_idx = slice(0, n_train), slice(n_train, n)

        delta = chosen_log - rejected_log
        delta_train, delta_test = delta[train_idx], delta[test_idx]

        if verbose:
            print(f"\n=== {user_id} ===")

        p, history = optimize_p(delta_train, epochs, lr, l1_lambda, device, beta)

        if verbose:
            print(f"p: {p}")
            for h in history:
                print(f"epoch {h['epoch']:03d} loss={h['loss']:.4f} "
                      f"nll={h['nll']:.4f} l1={h['l1']:.4f} train_acc={h['train_acc']:.4f}")

        acc = evaluate_accuracy(delta_test, p)
        
        if verbose:
            print(f"{user_id}: test_acc={acc:.4f} (n_train={n_train}, n_test={n - n_train})")
        
        return acc, p


def compute_average_accuracy(rewards, selected_attributes, train_ratio, epochs, lr, l1_lambda, 
                             device, beta, base_seed=None):
    """Compute average test accuracy across all users for given parameters."""
    all_acc = []
    for user_id in rewards.keys():
        try:
            if base_seed is not None:
                user_seed = hash(user_id) % (2**31) + base_seed
            else:
                user_seed = None
            
            acc, _ = evaluate_single_user(
                rewards, user_id, selected_attributes, train_ratio, epochs, lr, l1_lambda,
                device, beta, seed=user_seed, verbose=False
            )
            all_acc.append(acc)
        except Exception as e:
            print(f"Error processing {user_id}: {e}")
            continue
    
    if not all_acc:
        return 0.0, 0
    return float(np.mean(all_acc)), len(all_acc)


def run(args):
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rewards = load_rewards_dir(args.rewards_dir, args.num_users)

    if not rewards:
        print(f"No reward files found in {args.rewards_dir}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    all_acc = []

    for user_id in rewards.keys():
        acc, p = evaluate_single_user(
            rewards, user_id, args.selected_attributes, args.train_ratio, args.epochs, 
            args.lr, args.l1_lambda, device, args.beta, seed=None, verbose=True,
            leave_one_out=args.leave_one_out
        )
        
        results.append({
            "user": user_id,
            "p": p.tolist(),
        })
        all_acc.append(acc)

    if all_acc:
        mean_acc, std_acc = float(np.mean(all_acc)), float(np.std(all_acc))
        eval_mode = "leave-one-out" if args.leave_one_out else "train/test split"
        print(f"\nAvg accuracy across {len(all_acc)} users ({eval_mode}): {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("No users evaluated.")
    return 0

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
    400: list(range(400)),
    26: list(range(26))
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize drift vector p per user using gradient descent with L1.")
    parser.add_argument("--rewards_dir", type=str, default="eval_rewards/llama1b/prism/train")
    parser.add_argument("--num_users", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l1_lambda", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, required=True, help="Number of attributes to use (use preset indices)")
    parser.add_argument("--k_random", action="store_true",
                        help="If set, randomly select k attribute indices from the pool instead of preset list")
    parser.add_argument("--attr_pool_size", type=int, default=400,
                        help="Attribute index pool size to sample from when using --k_random (default: 400)")
    parser.add_argument("--mismatch_training", action="store_true",
                        help="If set, replace rejected responses with randomly selected chosen responses from other users")
    parser.add_argument("--leave_one_out", action="store_true",
                        help="If set, leave one out of the training set")
    parser.add_argument("--use_cuda", action="store_true",
                        help="If set, use CUDA")
    args = parser.parse_args()

    if args.k_random:
        if args.k > args.attr_pool_size:
            parser.error(f"--k ({args.k}) cannot exceed --attr_pool_size ({args.attr_pool_size})")
        rng = np.random.RandomState(args.seed)
        selected_attributes = rng.choice(np.arange(args.attr_pool_size), size=args.k, replace=False).tolist()
    else:
        selected_attributes = attributes.get(args.k, None)
        if selected_attributes is None:
            parser.error(f"--k={args.k} not found in preset attributes. Use --k_random to randomly select attributes.")
    
    args.selected_attributes = selected_attributes
    run(args)
    with open(f"selection_accuracy_grad.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
