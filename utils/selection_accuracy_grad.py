import os
import glob
import random as _random
import argparse
import numpy as np
import torch
import json

results = []

def compute_normalized_log_probs(rewards):
    eps = 1e-12

    selected_attributes = list(range(8))

    
    chosen_attr = rewards['attr_scores_chosen'][:, selected_attributes] / rewards['attr_counts_chosen'][:, selected_attributes].clamp(min=eps)
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
    """Bradley–Terry negative log-likelihood + L1."""
    logits = beta * (delta_phi @ p)
    nll = -torch.nn.functional.logsigmoid(logits).mean()
    l1_penalty = l1_lambda * torch.norm(p, 1)
    return nll + l1_penalty, nll, l1_penalty


def accuracy_from_delta(delta_phi: torch.Tensor, p: torch.Tensor) -> float:
    scores = delta_phi @ p
    correct = (scores > 0).sum().item()
    return correct / delta_phi.shape[0]

def optimize_p(delta_phi: torch.Tensor,
               epochs: int,
               lr: float,
               l1_lambda: float,
               device: torch.device,
               beta: float = 1.0):
    """Train p on delta_phi using BT + L1."""
    delta_phi = delta_phi.to(device)
    n, k = delta_phi.shape
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

    for user_id, r in rewards.items():
        chosen_log, rejected_log = compute_normalized_log_probs(r)

        if args.selected_attributes:
            idx = torch.tensor(args.selected_attributes, dtype=torch.long)
            chosen_log = chosen_log[:, idx].contiguous()
            rejected_log = rejected_log[:, idx].contiguous()

        n = chosen_log.shape[0]
        perm = torch.randperm(n)
        chosen_log = chosen_log[perm]
        rejected_log = rejected_log[perm]

        n_train = int(n * args.train_ratio)
        train_idx, test_idx = slice(0, n_train), slice(n_train, n)

        delta = chosen_log - rejected_log
        if args.swap_chosen_rejected:
            delta = -delta

        if args.random_sign_flip:
            mask = (torch.rand(delta.size(0)) > 0.5).float().unsqueeze(1)
            delta = delta * (1 - 2 * mask)

        if args.normalize_features:
            delta = (delta - delta.mean(0, keepdim=True)) / (delta.std(0, keepdim=True) + 1e-8)

        delta_train, delta_test = delta[train_idx], delta[test_idx]

        print(f"\n=== {user_id} ===")
        print("Fraction positive per attr (train):", (delta_train > 0).float().mean(0).cpu().numpy().round(3))
        print("Any negatives overall?", bool((delta_train < 0).any()))

        p, history = optimize_p(delta_train, args.epochs, args.lr, args.l1_lambda, device, args.beta)

        results.append({
            "user": user_id,
            "p": p.tolist(),
        })
        for h in history:
            print(f"epoch {h['epoch']:03d} loss={h['loss']:.4f} "
                  f"nll={h['nll']:.4f} l1={h['l1']:.4f} train_acc={h['train_acc']:.4f}")

        acc = evaluate_accuracy(delta_test, p)
        all_acc.append(acc)
        print(f"{user_id}: test_acc={acc:.4f} (n_train={n_train}, n_test={n - n_train})")

    if all_acc:
        mean_acc, std_acc = float(np.mean(all_acc)), float(np.std(all_acc))
        print(f"\nAvg accuracy across {len(all_acc)} users: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("No users evaluated.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize drift vector p per user using gradient descent with L1.")
    parser.add_argument("--rewards_dir", type=str, default="rewards_persona_testing")
    parser.add_argument("--num_users", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l1_lambda", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--selected_attributes", type=int, nargs="*", default=[])
    parser.add_argument("--swap_chosen_rejected", action="store_true",
                        help="Swap chosen/rejected to test symmetry.")
    parser.add_argument("--random_sign_flip", action="store_true",
                        help="Randomly flip 50%% of delta signs for sanity check.")
    parser.add_argument("--normalize_features", action="store_true",
                        help="Zero-mean, unit-variance normalize Δφ features.")
    args = parser.parse_args()
    if args.selected_attributes is None:
        args.selected_attributes = []

    run(args)
    with open(f"selection_accuracy_grad.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")