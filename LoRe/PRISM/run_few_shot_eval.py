#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List, Optional

import torch
import numpy as np

# Allow importing LoRe/utils.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LORE_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(LORE_ROOT)

from utils import run_few_shot_vary_shots  # noqa: E402
from PRISM.train_basis import group_embeddings_by_user  # noqa: E402


def parse_k_list(arg: str) -> List[int]:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",")]
    return [int(p) for p in parts if p]


def parse_int_list(arg: str) -> List[int]:
    return parse_k_list(arg)


def load_or_build_v_final(
    v_final_path: Optional[str],
    model_name: Optional[str],
    device: str,
    dtype: str,
    feature_dim_hint: Optional[int],
) -> torch.Tensor:
    if v_final_path:
        t = torch.load(v_final_path, map_location="cpu")
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return t.to(torch.float32)
        raise ValueError("--v_final_path must point to a 2D tensor (F, 1)")

    if not model_name:
        if feature_dim_hint is None:
            raise ValueError("Either --v_final_path or --model must be provided (no feature_dim_hint).")
        # Fallback: random unit vector with correct feature dim (not recommended)
        v = torch.randn(feature_dim_hint, 1)
        v = v / (v.norm() + 1e-8)
        return v.to(torch.float32)

    from transformers import AutoModel

    torch_dtype = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    # Extract last linear layer weight as in train_basis.py
    last_linear_layer = None
    for _, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module
    if last_linear_layer is None:
        raise RuntimeError("Could not find a Linear layer in the model to extract V_final.")
    v_final = last_linear_layer.weight[:, 0].to(torch.float32).reshape(-1, 1)
    return v_final


def main():
    ap = argparse.ArgumentParser(description="Run few-shot unseen evaluation on PRISM embeddings.")
    ap.add_argument("--train_embeddings", type=str, default="data/prism/train_embeddings.pkl")
    ap.add_argument("--test_embeddings", type=str, default="data/prism/test_embeddings.pkl")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])

    ap.add_argument("--v_final_path", type=str, default=None, help="Path to saved V_final tensor (F,1).")
    ap.add_argument("--model", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
                    help="HF model to extract V_final if --v_final_path not provided.")

    ap.add_argument("--alpha", type=float, default=1e4, help="Regularization strength.")
    ap.add_argument("--K_list", type=str, default="10", help="Comma-separated ranks, e.g. '0,10,20'.")
    ap.add_argument("--shots", type=str, default="1,2,5,10,20", help="Comma-separated few-shot counts.")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials per shots value.")
    args = ap.parse_args()

    # Load saved embeddings
    train_embeddings = torch.load(args.train_embeddings, map_location="cpu")
    test_embeddings = torch.load(args.test_embeddings, map_location="cpu")

    # Group per user -> list[tensor] where each tensor is (num_examples_user, feat_dim)
    train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(
        train_embeddings, test_embeddings, device=args.device
    )
    N, N_unseen = len(train_seen), len(train_unseen)
    if N == 0 or N_unseen == 0:
        print(f"Empty groups: N_seen={N}, N_unseen={N_unseen}. Check embeddings and splits.")
        sys.exit(1)

    # Infer feature dim for fallback V construction
    feat_dim = int(train_seen[0].shape[1])

    # Prepare V_final
    v_final = load_or_build_v_final(
        v_final_path=args.v_final_path,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        feature_dim_hint=feat_dim,
    )

    alpha_list = [float(args.alpha)]
    k_list = parse_k_list(args.K_list)
    num_shots = parse_int_list(args.shots)

    few_mean, few_std, unseen_mean, unseen_std = run_few_shot_vary_shots(
        trials=int(args.trials),
        alpha_list=alpha_list,
        K_list=k_list,
        num_shots=num_shots,
        train_features=train_seen,
        train_features_unseen=train_unseen,
        test_features_sparse_unseen=test_unseen,
        V_final=v_final,
        N=N,
        N_unseen=N_unseen,
        device=args.device,
    )

    print("Few-shot train (means):", np.array(few_mean))
    print("Few-shot train (stds):", np.array(few_std))
    print("Unseen accuracy (means):", np.array(unseen_mean))
    print("Unseen accuracy (stds):", np.array(unseen_std))

    # Optional: save a plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        x = num_shots
        y = unseen_mean
        yerr = unseen_std
        plt.errorbar(x, y, yerr=yerr, marker='o', capsize=3)
        plt.xlabel("Few-shot samples")
        plt.ylabel("Unseen accuracy")
        plt.title(f"Few-shot unseen accuracy vs shots (trials={args.trials}, alpha={args.alpha})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, "fewshot_unseen_accuracy.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    except Exception as e:
        print(f"Plotting failed (skipping): {e}")


if __name__ == "__main__":
    main()


