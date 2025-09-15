#!/usr/bin/env python3
"""
Learner Server: Handles model training and parameter updates.
- Runs on a specified GPU (default cuda:1)
- Receives batches (reward matrices) via HTTP and updates model
- Optimized to keep FastAPI event loop responsive:
    * Heavy Torch ops run in a worker thread (anyio.to_thread)
    * TF32/AMP enabled for speed on modern GPUs
    * Checkpoint I/O offloaded to thread as well
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

# Optional: faster event loop on Linux/macOS
try:
    import uvloop  # type: ignore
    UVLOOP_AVAILABLE = True
except Exception:
    UVLOOP_AVAILABLE = False

# Optional: Weights & Biases
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

import anyio

# ---- Local imports (adjust to your repo layout) ----
from ..models.sparse_mask import SparseMaskModel  # ensure this exposes forward() returning (z, x_hat, masks)

# =========================
# Request/Response models
# =========================
class ParametersResponse(BaseModel):
    mask_logits: List[float]
    step: int
    tau: float
    success: bool
    error: Optional[str] = None

class TrainStepRequest(BaseModel):
    m_hard: List[float]                 # if unused for loss, consider removing to save bandwidth
    R: List[List[float]]               # [batch_size, d] reward matrix
    user_data: Dict[str, Any]          # passthrough (for logging/analysis)
    success: bool
    error: Optional[str] = None

class TrainStepResponse(BaseModel):
    success: bool
    step: int
    loss: float
    reward_signal: float
    active_attributes: float
    error: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    device: str
    current_step: int
    active_features: float

# =========================
# Globals
# =========================
app = FastAPI(title="Learner Server", version="1.1")
app.add_middleware(GZipMiddleware, minimum_size=1024)

model: Optional[SparseMaskModel] = None
optimizer: Optional[torch.optim.Optimizer] = None
device: Optional[torch.device] = None
current_step: int = 0
tau: float = 1.0
wandb_run = None
checkpoint_dir: str = "./checkpoints"
checkpoint_every: int = 500  # steps between checkpoints (configurable)
training_lock = asyncio.Lock()  # ensure only one train step mutates model at a time

# Scalars for saving config (avoid relying on model.d/model.k existence)
D_SIZE: int = 0
K_SIZE: int = 0
SPARSITY_WEIGHT: float = 0.1

# AMP/TF32 helpers
USE_AMP: bool = False
grad_scaler: Optional[torch.cuda.amp.GradScaler] = None

# =========================
# Initialization
# =========================
def initialize_learner(
    d: int,
    k: int,
    lr: float,
    sparsity_weight: float,
    tau_init: float,
    device_str: str,
    checkpoint_dir_arg: str,
    use_wandb: bool,
    ckpt_every: int,
):
    """Initialize learner components (sync)."""
    global model, optimizer, device, tau, wandb_run, checkpoint_dir
    global D_SIZE, K_SIZE, SPARSITY_WEIGHT, USE_AMP, grad_scaler, checkpoint_every

    device = torch.device(device_str)
    tau = tau_init
    checkpoint_dir = checkpoint_dir_arg
    checkpoint_every = ckpt_every

    # Enable TF32/AMP where available
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    USE_AMP = device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    logging.info("Initializing learner")
    logging.info("Device: %s", device)
    logging.info("Checkpoint directory: %s", checkpoint_dir)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model/optimizer
    model = SparseMaskModel(d, k, sparsity_weight=sparsity_weight).to(device)  # assumes constructor signature
    # Persist size params in case model doesn't expose attributes
    D_SIZE = d
    K_SIZE = k
    SPARSITY_WEIGHT = sparsity_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Resume if possible
    if load_latest_checkpoint():
        logging.info("Resumed training from checkpoint at step %d", current_step)
    else:
        logging.info("Starting training from scratch")

    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project="sparse-attributes-distributed",
                config={
                    "d": d, "k": k, "lr": lr, "sparsity_weight": sparsity_weight,
                    "tau_init": tau_init, "device": device_str,
                    "checkpoint_every": checkpoint_every,
                }
            )
            logging.info("W&B initialized")
        except Exception as e:
            logging.error("Failed to initialize W&B: %s", e)

    logging.info("Learner ready: d=%d -> k=%d", d, k)

# =========================
# Core training (sync)
# =========================
def train_step_impl(m_hard: torch.Tensor, R: torch.Tensor, step: int) -> Dict[str, float]:
    """Single training step using reward matrix as input. Runs under a lock in a worker thread."""
    assert model is not None and optimizer is not None and device is not None
    assert grad_scaler is not None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Normalize input per-sample like original
    x = F.normalize(R, p=2, dim=1)

    # Forward + loss
    if USE_AMP:
        with torch.cuda.amp.autocast():
            z, x_hat, masks = model.forward(x)  # assumes returns (z, x_hat, masks)
            recon_loss = F.mse_loss(x_hat, x)
            mask_probs = torch.sigmoid(model.mask_logits)
            sparsity_loss = mask_probs.mean()
            loss = recon_loss + model.sparsity_weight * sparsity_loss
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        z, x_hat, masks = model.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        mask_probs = torch.sigmoid(model.mask_logits)
        sparsity_loss = mask_probs.mean()
        loss = recon_loss + model.sparsity_weight * sparsity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "reward_signal": float(recon_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "avg_reward": float(R.mean().detach().cpu()) if R.numel() > 0 else 0.0,
        "active_attributes": float(masks.sum().detach().cpu()),
    }
    return metrics

# =========================
# Checkpointing (sync)
# =========================
def save_checkpoint(step: int, final: bool = False) -> Optional[str]:
    """Save model checkpoint (sync)."""
    try:
        assert model is not None and optimizer is not None
        # move weights to CPU for lighter write and to avoid GPU stalls
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        checkpoint = {
            "model_state_dict": cpu_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "tau": tau,
            "model_config": {
                "d": D_SIZE,
                "k": K_SIZE,
                "sparsity_weight": SPARSITY_WEIGHT
            }
        }

        if final:
            path = Path(checkpoint_dir) / "final_checkpoint.pt"
        else:
            path = Path(checkpoint_dir) / "latest.pt"

        torch.save(checkpoint, path)
        logging.info("Saved checkpoint: %s (step %d)", str(path), step)
        return str(path)
    except Exception as e:
        logging.error("Failed to save checkpoint: %s", e)
        return None

def load_checkpoint(checkpoint_path: str) -> bool:
    """Load model/optimizer from checkpoint (sync)."""
    global current_step, tau
    try:
        if not Path(checkpoint_path).exists():
            logging.info("Checkpoint not found: %s", checkpoint_path)
            return False

        assert model is not None and optimizer is not None and device is not None

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        current_step = int(checkpoint.get("step", 0))
        tau = float(checkpoint.get("tau", 1.0))

        cfg = checkpoint.get("model_config", {})
        if isinstance(cfg, dict):
            # Not strictly necessary; kept for logging sanity
            logging.info(
                "Loaded checkpoint cfg: d=%s k=%s sparsity_weight=%s",
                cfg.get("d", "?"), cfg.get("k", "?"), cfg.get("sparsity_weight", "?")
            )

        logging.info("Loaded checkpoint from %s: step=%d tau=%.4f", checkpoint_path, current_step, tau)
        return True
    except Exception as e:
        logging.error("Failed to load checkpoint %s: %s", checkpoint_path, e)
        return False

def load_latest_checkpoint() -> bool:
    latest_path = Path(checkpoint_dir) / "latest.pt"
    return load_checkpoint(str(latest_path))

# =========================
# Endpoints
# =========================
@app.get("/get_params", response_model=ParametersResponse)
async def get_params():
    """Alias: returns current mask logits, step, and tau."""
    return await get_parameters()

@app.get("/parameters", response_model=ParametersResponse)
async def get_parameters():
    """Get current model parameters (for collector's behavior policy)."""
    try:
        if model is None:
            return ParametersResponse(
                mask_logits=[], step=0, tau=0.0, success=False, error="Model not initialized"
            )
        mask_logits = model.mask_logits.detach().cpu().tolist()
        return ParametersResponse(mask_logits=mask_logits, step=current_step, tau=tau, success=True)
    except Exception as e:
        return ParametersResponse(mask_logits=[], step=0, tau=0.0, success=False, error=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get learner status."""
    active_features = 0.0
    try:
        if model is not None:
            active_features = float(torch.sigmoid(model.mask_logits).sum().item())
    except Exception:
        active_features = 0.0

    return StatusResponse(
        status="running" if model is not None else "initializing",
        device=str(device) if device else "unknown",
        current_step=current_step,
        active_features=active_features
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_ready": model is not None}

@app.post("/save_checkpoint")
async def save_checkpoint_endpoint():
    """Save checkpoint asynchronously (does not block event loop)."""
    path = await anyio.to_thread.run_sync(save_checkpoint, current_step, False)
    if path:
        return {"success": True, "checkpoint_path": path, "step": current_step}
    return {"success": False, "error": "Failed to save checkpoint"}

@app.post("/save_final_checkpoint")
async def save_final_checkpoint_endpoint():
    """Save final checkpoint asynchronously."""
    path = await anyio.to_thread.run_sync(save_checkpoint, current_step, True)
    if path:
        return {"success": True, "checkpoint_path": path, "step": current_step}
    return {"success": False, "error": "Failed to save final checkpoint"}

@app.post("/train_step", response_model=TrainStepResponse)
async def train_step_endpoint(request: TrainStepRequest):
    """
    Perform a single training step with provided batch data.
    Heavy compute runs in a worker thread; endpoint remains async.
    """
    global current_step, tau

    try:
        if model is None or optimizer is None or device is None:
            return TrainStepResponse(
                success=False, step=current_step, loss=0.0,
                reward_signal=0.0, active_attributes=0.0,
                error="Model not initialized"
            )

        # Basic validation
        if request.R is None or len(request.R) == 0:
            return TrainStepResponse(
                success=False, step=current_step, loss=0.0,
                reward_signal=0.0, active_attributes=0.0,
                error="Empty reward matrix"
            )

        R_tensor = torch.tensor(request.R, dtype=torch.float32)  # on CPU first
        if R_tensor.ndim != 2:
            return TrainStepResponse(
                success=False, step=current_step, loss=0.0,
                reward_signal=0.0, active_attributes=0.0,
                error="R must be 2D [B, d]"
            )

        B, d = int(R_tensor.shape[0]), int(R_tensor.shape[1])
        if D_SIZE and d != D_SIZE:
            return TrainStepResponse(
                success=False, step=current_step, loss=0.0,
                reward_signal=0.0, active_attributes=0.0,
                error=f"R has d={d}, expected d={D_SIZE}"
            )

        m_hard_tensor = torch.tensor(request.m_hard, dtype=torch.float32) if request.m_hard else torch.zeros(d)

        # Move to device inside worker to avoid blocking loop
        async with training_lock:
            # Offload training to worker thread
            def _train():
                m = m_hard_tensor.to(device, non_blocking=True)
                Rt = R_tensor.to(device, non_blocking=True)
                metrics = train_step_impl(m, Rt, current_step)
                return metrics

            metrics = await anyio.to_thread.run_sync(_train)

            current_step += 1

            # Temperature annealing
            if current_step % 100 == 0 and current_step > 0:
                tau_new = tau * 0.995
                tau_value = 0.1 if tau_new < 0.1 else tau_new
                # assign outside of any potential autocast context
                tau = float(tau_value)

            # Periodic checkpointing
            if checkpoint_every > 0 and current_step % checkpoint_every == 0:
                # fire-and-forget checkpoint write (but await here to keep it simple & safe)
                await anyio.to_thread.run_sync(save_checkpoint, current_step, False)

        # Wandb logging (lightweight)
        if WANDB_AVAILABLE and wandb_run:
            try:
                active_features = float(torch.sigmoid(model.mask_logits).sum().item()) if model is not None else 0.0
                wandb_run.log({
                    **metrics,
                    "step": current_step,
                    "active_features": active_features,
                    "temperature": tau,
                    "batch_size": B,
                }, step=current_step)
            except Exception as e:
                logging.warning("wandb logging failed: %s", e)

        return TrainStepResponse(
            success=True,
            step=current_step,
            loss=metrics["loss"],
            reward_signal=metrics["reward_signal"],
            active_attributes=metrics["active_attributes"]
        )

    except Exception as e:
        logging.exception("Training step failed")
        return TrainStepResponse(
            success=False, step=current_step, loss=0.0,
            reward_signal=0.0, active_attributes=0.0,
            error=str(e)
        )

@app.on_event("shutdown")
async def shutdown_event():
    """Save final checkpoint and close wandb on shutdown."""
    logging.info("Shutting down learner server, saving final checkpoint...")
    try:
        await anyio.to_thread.run_sync(save_checkpoint, current_step, True)
    except Exception as e:
        logging.error("Final checkpoint save failed: %s", e)
    if WANDB_AVAILABLE and wandb_run:
        try:
            wandb_run.finish()
            logging.info("Finished wandb run")
        except Exception as e:
            logging.warning("Failed to finish wandb run: %s", e)

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Learner Server")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    
    # Individual parameter overrides (for backward compatibility)
    parser.add_argument("--d", type=int, help="Number of attributes")
    parser.add_argument("--k", type=int, help="Number of components")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, help="Sparsity weight")
    parser.add_argument("--tau-init", type=float, help="Initial temperature")
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--device", type=str, help="Device for learner")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, help="Checkpoint every N steps")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--log-level", type=str, help="Log level")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        try:
            from ..utils.config_loader import load_config, ConfigLoader
            config = load_config(args.config)
            learner_config = ConfigLoader.get_learner_config(config)
            
            # Apply config values as defaults, allow CLI overrides
            d = args.d or learner_config["d"]
            k = args.k or learner_config["k"]
            lr = args.lr or learner_config["lr"]
            sparsity_weight = args.sparsity_weight or learner_config["sparsity_weight"]
            tau_init = args.tau_init or learner_config["tau_init"]
            host = args.host or learner_config["host"]
            port = args.port or learner_config["port"]
            device_str = args.device or learner_config["device"]
            checkpoint_dir_arg = args.checkpoint_dir or learner_config["checkpoint_dir"]
            checkpoint_every = args.checkpoint_every or learner_config["checkpoint_every"]
            use_wandb = args.use_wandb or learner_config["use_wandb"]
            log_level = args.log_level or learner_config["log_level"]
            
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logging.info(f"Loaded learner config from {args.config}")
        except Exception as e:
            logging.error(f"Failed to load config from {args.config}: {e}")
            return
    else:
        # Use command line arguments with defaults
        d = args.d or 100
        k = args.k or 10
        lr = args.lr or 1e-3
        sparsity_weight = args.sparsity_weight or 0.1
        tau_init = args.tau_init or 1.0
        host = args.host or "0.0.0.0"
        port = args.port or 8002
        device_str = args.device or "cuda:1"
        checkpoint_dir_arg = args.checkpoint_dir or "./checkpoints"
        checkpoint_every = args.checkpoint_every or 500
        use_wandb = args.use_wandb
        log_level = args.log_level or "INFO"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Install uvloop if available
    if UVLOOP_AVAILABLE:
        try:
            uvloop.install()
        except Exception:
            pass

    # Initialize learner synchronously on startup
    @app.on_event("startup")
    async def startup_event():
        initialize_learner(
            d=d,
            k=k,
            lr=lr,
            sparsity_weight=sparsity_weight,
            tau_init=tau_init,
            device_str=device_str,
            checkpoint_dir_arg=checkpoint_dir_arg,
            use_wandb=use_wandb,
            ckpt_every=checkpoint_every,
        )

    logging.info("Starting Learner Server on %s:%d", host, port)
    logging.info("Device: %s", device_str)
    logging.info("Model: d=%d -> k=%d", d, k)

    # Important: use a single process/worker (GPU state is not multiprocess-safe)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        workers=1
    )

if __name__ == "__main__":
    main()
