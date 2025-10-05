#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import json
import asyncio
import argparse
import yaml
from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
import wandb

# --- Your CAE layers / Lightning wrapper ---
from cae import ConcreteLinear
from pl_wrappers import PL_CAE_Wrapper


# ---------------------------
#   Model (unchanged API)
# ---------------------------
class BatchedConcreteLinear(ConcreteLinear):
    def __init__(self, input_dim: int, *args, **kwargs):
        # ConcreteLinear expects input_dim and optional mask_ratio or k
        super().__init__(input_dim=input_dim, *args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Pass through training-time kwargs like random, temperature, hard, etc.
        return super().forward(x, **kwargs)


# -----------------------------------------
#   Samplers and On-the-fly dataset
# -----------------------------------------

def make_persona_pref_sampler(
    data_root: str,
    users_per_batch: int,
    pairs_per_user: int,
    gateway_url: str,
    tokenizer,
    base_prompt: str,
    attribute_prompts: List[str],
    model_name: str,
    device: Optional[torch.device] = None,
):
    """
    Build a sampler that, on each call, samples users in [11, 200], loads their
    persona_pref train files, samples preference pairs, and computes on-the-fly
    drift rewards using the VLLM gateway (see src/core/drift.compute_rewards).

    Returns: sample_fn() -> torch.FloatTensor of shape [B, D]
      where B = users_per_batch * pairs_per_user, D = len(attribute_prompts).

    The batch value is R_chosen - R_rejected per pair.
    """

    # Local import to avoid import-time heavy deps for environments not using this
    from drift_helper import compute_rewards

    dev = device if device is not None else torch.device("cpu")
    d = len(attribute_prompts)
    rng = np.random.RandomState()

    persona_dir = os.path.join(data_root, "persona_pref")

    def _load_user_train(u: int) -> List[Dict]:
        path = os.path.join(persona_dir, f"user{u}_train.json")
        with open(path, "r") as f:
            return json.load(f)

    def sample_fn() -> torch.Tensor:
        # 1) Choose users in inclusive range [11, 200]
        users = rng.choice(np.arange(11, 201), size=users_per_batch, replace=True)

        prompts: List[str] = []
        chosen_list: List[str] = []
        rejected_list: List[str] = []

        # 2) For each user, load train file and sample pairs
        for u in users:
            data = _load_user_train(int(u))  # records with keys: prompt, chosen, rejected
            n = len(data)
            if n == 0:
                continue
            replace = pairs_per_user > n
            idx = rng.choice(np.arange(n), size=pairs_per_user, replace=replace)
            for i in idx:
                rec = data[int(i)]
                prompts.append(rec["prompt"])
                chosen_list.append(rec["chosen"])
                rejected_list.append(rec["rejected"])

        B = len(prompts)
        if B == 0:
            return torch.zeros(0, d, device=dev, dtype=torch.float32)

        # 3) Compute rewards on-the-fly for chosen and rejected with the gateway
        async def _compute_batch():
            # Launch chosen and rejected concurrently to maximize gateway utilization
            Rc_task = compute_rewards(
                gateway_url=gateway_url,
                tokenizer=tokenizer,
                prompts=prompts,
                outputs=chosen_list,
                base_prompt=base_prompt,
                attribute_prompts=attribute_prompts,
                model_name=model_name,
                device=torch.device('cpu'),
            )
            Rr_task = compute_rewards(
                gateway_url=gateway_url,
                tokenizer=tokenizer,
                prompts=prompts,
                outputs=rejected_list,
                base_prompt=base_prompt,
                attribute_prompts=attribute_prompts,
                model_name=model_name,
                device=torch.device('cpu'),
            )
            R_chosen, R_rejected = await asyncio.gather(Rc_task, Rr_task)
            # Drop rows that failed (all-zero or NaN rows)
            def _valid_rows(M: torch.Tensor) -> torch.Tensor:
                if M.numel() == 0:
                    return torch.zeros(0, dtype=torch.bool)
                nan_bad = torch.isnan(M).any(dim=1)
                zero_bad = (M.abs().sum(dim=1) == 0)
                return ~(nan_bad | zero_bad)
            vc = _valid_rows(R_chosen)
            vr = _valid_rows(R_rejected)
            keep = vc & vr
            if keep.numel() == 0 or keep.sum().item() == 0:
                return torch.zeros(0, R_chosen.shape[1], dtype=R_chosen.dtype)
            return (R_chosen - R_rejected)[keep]

        try:
            loop = asyncio.get_running_loop()
            X = loop.run_until_complete(_compute_batch())  # type: ignore[attr-defined]
        except RuntimeError:
            X = asyncio.run(_compute_batch())

        return X.cpu()

    # Allow DataLoader workers to reseed the RNG
    def reseed(seed: int):
        nonlocal rng
        rng = np.random.RandomState(seed)

    sample_fn.reseed = reseed  # type: ignore[attr-defined]
    sample_fn.feature_dim = d  # type: ignore[attr-defined]
    return sample_fn


class ReplayBuffer:
    """Tiny, optional replay buffer to reuse past samples cheaply."""
    def __init__(self, capacity: int, feature_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False
        self.buf = torch.empty((capacity, feature_dim), dtype=torch.float32, device=device)

    def add(self, x: torch.Tensor):
        b = x.shape[0]
        end = self.ptr + b
        if end <= self.capacity:
            self.buf[self.ptr:end] = x
        else:
            first = self.capacity - self.ptr
            self.buf[self.ptr:] = x[:first]
            self.buf[: end % self.capacity] = x[first:]
        self.ptr = end % self.capacity
        if b >= self.capacity or self.ptr == 0:
            self.full = True

    def sample(self, k: int) -> torch.Tensor:
        size = self.capacity if self.full else max(self.ptr, 1)
        idx = torch.randint(0, size, (k,), device=self.device)
        return self.buf.index_select(0, idx)


class OnTheFlyDataset(IterableDataset):
    """
    Iterable dataset yielding (X, y) where X is produced by sample_fn().
    y is a dummy tensor (ignored by reconstruction models).
    """
    def __init__(
        self,
        sample_fn: Callable[[], torch.Tensor],
        steps_per_epoch: int,
        replay: Optional[ReplayBuffer] = None,
        replay_ratio: float = 0.0,  # fraction of batch from replay (0..1)
        batch_size_hint: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.sample_fn = sample_fn
        self.steps_per_epoch = int(steps_per_epoch)
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.dtype = dtype
        self._batch_size_hint = batch_size_hint  # optional; used only for replay mixing

    def __len__(self):
        # Helps Lightning progress bars & sanity checks
        return self.steps_per_epoch

    def _seed_worker(self):
        info = get_worker_info()
        if info is None:
            # main process
            seed = torch.initial_seed() % (2**31)
            if hasattr(self.sample_fn, "reseed"):
                self.sample_fn.reseed(int(seed))
        else:
            # distinct seed per worker
            base_seed = torch.initial_seed() % (2**31)
            if hasattr(self.sample_fn, "reseed"):
                self.sample_fn.reseed(int(base_seed + info.id))

    def __iter__(self):
        self._seed_worker()

        for _ in range(self.steps_per_epoch):
            X = self.sample_fn().to(dtype=self.dtype)

            # Fill replay (before mixing)
            if self.replay is not None:
                self.replay.add(X)

            if self.replay is not None and self.replay_ratio > 0.0 and self._batch_size_hint:
                k_replay = int(self._batch_size_hint * self.replay_ratio)
                if k_replay > 0:
                    X_re = self.replay.sample(k_replay)
                    X = torch.cat([X, X_re], dim=0)

            # Dummy labels (ignored by wrapper)
            y = torch.zeros(X.size(0), 1, device=X.device, dtype=X.dtype)
            yield (X, y)


def make_on_the_fly_loader(
    sample_fn: Callable[[], torch.Tensor],
    steps_per_epoch: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    replay: Optional[ReplayBuffer] = None,
    replay_ratio: float = 0.0,
    batch_size_hint: Optional[int] = None,
) -> DataLoader:
    ds = OnTheFlyDataset(
        sample_fn=sample_fn,
        steps_per_epoch=steps_per_epoch,
        replay=replay,
        replay_ratio=replay_ratio,
        batch_size_hint=batch_size_hint,
    )
    return DataLoader(
        ds,
        batch_size=None,           # dataset yields pre-batched tensors
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


# -----------------------------------------
#   Lightning wrapper
# -----------------------------------------
class PL_Batched_CAE(PL_CAE_Wrapper):
    """
    Lightning wrapper that samples training data on-the-fly via `sample_fn`.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        args,  # your args object, must include dim_ip (feature dim)
        sample_fn: Callable[[], torch.Tensor],
        steps_per_epoch: int,
        use_replay: bool = False,
        replay_capacity: int = 0,
        replay_ratio: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        # Provide dummy datasets; we'll override train_dataloader().
        super().__init__(model=model, args=args, datasets=(None, None, None))

        # Optional replay buffer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        replay = None
        batch_size_hint = None
        if use_replay and replay_capacity > 0:
            feat_dim = getattr(sample_fn, "feature_dim", getattr(args, "dim_ip", None))
            if feat_dim is None:
                raise ValueError("Cannot infer feature_dim for ReplayBuffer.")
            replay = ReplayBuffer(replay_capacity, feat_dim, device=device)
            # We can try to infer batch size by a dry call
            with torch.no_grad():
                _x = sample_fn()
                batch_size_hint = int(_x.size(0))
                del _x

        self._train_loader = make_on_the_fly_loader(
            sample_fn=sample_fn,
            steps_per_epoch=steps_per_epoch,
            num_workers=num_workers if hasattr(self, "num_workers") is False else self.num_workers,
            pin_memory=pin_memory if hasattr(self, "pin_mem") is False else self.pin_mem,
            replay=replay,
            replay_ratio=replay_ratio,
            batch_size_hint=batch_size_hint,
        )
        self._val_dataset = val_dataset

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def on_train_epoch_end(self):
        try:
            model = self.model
            gd = model.gumbel_distrib
            gd.eval()
            model.eval()
            with torch.no_grad():
                L = gd.get_logits()          # (K, D)
                idx = L.argmax(dim=1)        # (K,)
            selected = idx.detach().cpu().tolist()
            print(f"[Epoch {self.current_epoch}] Selected features (K={len(selected)}): {selected}")
            # Optionally log to wandb
            try:
                import numpy as _np
                self.log_dict({"selected_feature_sample": float(selected[0]) if len(selected)>0 else -1}, on_epoch=True)
                if wandb.run is not None:
                    wandb.log({"selected_features/epoch": self.current_epoch, "selected_features/indices": selected})
            except Exception:
                pass
        except Exception as e:
            print(f"[Epoch {self.current_epoch}] Failed to extract selected features: {e}")

    def val_dataloader(self):
        if self._val_dataset is None:
            return []
        return DataLoader(
            self._val_dataset,
            batch_size=self.test_batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )


# -----------------------------------------
#   Example usage / sanity run
# -----------------------------------------
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # speedup if available

    parser = argparse.ArgumentParser(description="On-the-fly CAE training over persona_pref with attribute prompts")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rewards_example.yaml"))
    parser.add_argument("--attribute-config", type=str, default=None)
    parser.add_argument("--gateway-url", type=str, default=os.environ.get("GATEWAY_URL", "http://localhost:8080"))
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--base-prompt", type=str, default=None)
    parser.add_argument("--users-per-batch", type=int, default=None)
    parser.add_argument("--pairs-per-user", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    args_cli = parser.parse_args()

    # Load YAML config
    with open(args_cli.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    # Resolve paths and settings from config
    data_root = cfg_yaml.get("data_root_dir", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))
    k_sel = int(cfg_yaml.get("k", 20))
    epochs = int(cfg_yaml.get("epochs", 1))
    num_workers = int(cfg_yaml.get("num_workers", 0))
    dim_ip = int(cfg_yaml.get("dim_ip", 400))
    wandb_project = cfg_yaml.get("wandb", "ip-cae-batched")

    # Optional runtime batch composition (allow override via CLI)
    users_per_batch = args_cli.users_per_batch if args_cli.users_per_batch is not None else int(cfg_yaml.get("users_per_batch", 8))
    pairs_per_user = args_cli.pairs_per_user if args_cli.pairs_per_user is not None else int(cfg_yaml.get("pairs_per_user", 4))
    steps_per_epoch = args_cli.steps_per_epoch if args_cli.steps_per_epoch is not None else int(cfg_yaml.get("steps_per_epoch", 50))

    # Attribute prompts file
    attr_cfg_path = args_cli.attribute_config or cfg_yaml.get("attribute_config", os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "attribute_prompts_400.json"))
    with open(attr_cfg_path, "r") as f:
        cfg = json.load(f)
    if isinstance(cfg, dict) and "prompts" in cfg:
        attribute_prompts = cfg["prompts"]
    elif isinstance(cfg, list):
        attribute_prompts = cfg
    else:
        raise ValueError("attribute_config must be a list or a dict with key 'prompts'")
    D = len(attribute_prompts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args_cli.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build on-the-fly sampler that computes rewards via VLLM gateway
    sample_fn = make_persona_pref_sampler(
        data_root=data_root,
        users_per_batch=users_per_batch,
        pairs_per_user=pairs_per_user,
        gateway_url=args_cli.gateway_url,
        tokenizer=tokenizer,
        base_prompt=(args_cli.base_prompt or cfg_yaml.get("base_prompt", "You are an AI assistant.")),
        attribute_prompts=attribute_prompts,
        model_name=args_cli.model_name,
        device=device,
    )

    # Minimal args for PL_CAE_Wrapper
    class Args:
        dim_ip = D
        batch_size = users_per_batch * pairs_per_user
        test_batch_size = batch_size
        num_workers = cfg_yaml.get("num_workers", 0)
        pin_mem = int(device.type == "cuda")
        weight_decay = float(cfg_yaml.get("weight_decay", 0.0))
        jsd_factor = float(cfg_yaml.get("jsd_factor", 0.0))
        eeg_factor = float(cfg_yaml.get("eeg_factor", 0.0))
        loss_type = cfg_yaml.get("loss_type", "mse")
        temp_base = float(cfg_yaml.get("temp_base", 1.0))
        thresh_base = float(cfg_yaml.get("thresh_base", 0.5))
        rao_samples = int(cfg_yaml.get("rao_samples", 0))
        straight_through = bool(cfg_yaml.get("straight_through", 1))
        use_masked_loss = int(cfg_yaml.get("use_masked_loss", 0))
        lr = float(cfg_yaml.get("lr", 1e-3))
        norm_pix_loss = int(cfg_yaml.get("norm_pix_loss", 0))
        local_logging = int(cfg_yaml.get("local_logging", 0))

    args_stub = Args()

    # Concrete AE model (linear decoder) with k selected features
    model = BatchedConcreteLinear(input_dim=D, k=k_sel, dim_ip=D).to(device)

    # Build a small, deterministic validation set from 10 users' *_valid.json
    # Each record: (prompt, chosen, rejected) -> we form X = reward(chosen) - reward(rejected)
    class ValSet(torch.utils.data.Dataset):
        def __init__(self, data_root: str, tokenizer, base_prompt: str, attribute_prompts: List[str], model_name: str, users: int = 10, pairs_per_user: int = 2):
            self.X: List[torch.Tensor] = []
            persona_dir = os.path.join(data_root, "persona_pref")
            rng = np.random.RandomState(123)
            # Sample users 11..200
            user_ids = rng.choice(np.arange(11, 201), size=users, replace=False)
            # Load and sample pairs per user
            pairs: List[Dict] = []
            for u in user_ids:
                p = os.path.join(persona_dir, f"user{int(u)}_valid.json")
                if not os.path.exists(p):
                    continue
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if not data:
                    continue
                idx = rng.choice(np.arange(len(data)), size=min(pairs_per_user, len(data)), replace=False)
                for i in idx:
                    pairs.append(data[int(i)])

            # Compute rewards on CPU in a few mini-batches to avoid long serial calls
            from drift_helper import compute_rewards

            BATCH = 16
            for start in range(0, len(pairs), BATCH):
                chunk = pairs[start:start + BATCH]
                if not chunk:
                    continue
                prompts = [c["prompt"] for c in chunk]
                chosen = [c["chosen"] for c in chunk]
                rejected = [c["rejected"] for c in chunk]

                async def _comp():
                    Rc = await compute_rewards(
                        gateway_url=args_cli.gateway_url,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        outputs=chosen,
                        base_prompt=base_prompt,
                        attribute_prompts=attribute_prompts,
                        model_name=args_cli.model_name,
                        device=torch.device('cpu'),
                    )
                    Rr = await compute_rewards(
                        gateway_url=args_cli.gateway_url,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        outputs=rejected,
                        base_prompt=base_prompt,
                        attribute_prompts=attribute_prompts,
                        model_name=args_cli.model_name,
                        device=torch.device('cpu'),
                    )
                    return Rc - Rr

                try:
                    loop = asyncio.get_running_loop()
                    Xb = loop.run_until_complete(_comp())  # type: ignore[attr-defined]
                except RuntimeError:
                    Xb = asyncio.run(_comp())
                self.X.append(Xb.cpu())

            self.X = [x.to(torch.float32) for x in self.X]
            self.X = torch.cat(self.X, dim=0) if self.X else torch.zeros(0, len(attribute_prompts))

        def __len__(self):
            return self.X.size(0)

        def __getitem__(self, idx):
            x = self.X[idx]
            y = torch.zeros(1, dtype=torch.float32)
            return x, y

    base_prompt = (args_cli.base_prompt or cfg_yaml.get("base_prompt", "You are an AI assistant."))
    val_ds = ValSet(data_root, tokenizer, base_prompt, attribute_prompts, args_cli.model_name, users=10, pairs_per_user=2)

    pl_module = PL_Batched_CAE(
        model=model,
        args=args_stub,
        sample_fn=sample_fn,
        steps_per_epoch=args_cli.steps_per_epoch,
        use_replay=False,
        replay_capacity=0,
        replay_ratio=0.0,
        num_workers=cfg_yaml.get("num_workers", 0),
        pin_memory=False,
        val_dataset=val_ds if len(val_ds) > 0 else None,
    )

    # Init Weights & Biases logger (ensures wandb is initiated)
    wandb_logger = WandbLogger(project=os.environ.get("WANDB_PROJECT", wandb_project))

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        logger=wandb_logger,
    )

    trainer.fit(pl_module)
