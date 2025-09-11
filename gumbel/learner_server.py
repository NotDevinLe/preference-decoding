#!/usr/bin/env python3
"""
Learner Server: Handles model training and parameter updates.
Runs on GPU 1, requests data from collector server via HTTP.
"""

import asyncio
import json
import logging
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Local imports
from skeleton import SparseMaskModel

# Wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Request/Response models for new distributed architecture
class ParametersResponse(BaseModel):
    mask_logits: List[float]
    step: int
    tau: float
    success: bool
    error: str = None

class TrainStepRequest(BaseModel):
    m_hard: List[float]
    R: List[List[float]]  # [batch_size, d] reward matrix
    user_data: Dict[str, Any]
    success: bool
    error: str = None

class TrainStepResponse(BaseModel):
    success: bool
    step: int
    loss: float
    reward_signal: float
    active_attributes: float
    error: str = None

class StatusResponse(BaseModel):
    status: str
    device: str
    current_step: int
    active_features: float

# Global variables
app = FastAPI(title="Learner Server", version="1.0")
model = None
optimizer = None
device = None
current_step = 0
tau = 1.0
wandb_run = None

def initialize_learner(d: int, k: int, lr: float, sparsity_weight: float,
                      tau_init: float, device_str: str, 
                      checkpoint_dir: str, use_wandb: bool):
    """Initialize learner components"""
    global model, optimizer, device, tau, wandb_run
    
    device = torch.device(device_str)
    tau = tau_init
    
    logging.info(f"Initializing learner on {device}")
    
    # Initialize model
    model = SparseMaskModel(d, k, sparsity_weight=sparsity_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project="sparse-attributes-distributed",
                config={
                    "d": d, "k": k, "lr": lr, "sparsity_weight": sparsity_weight,
                    "tau_init": tau_init, "device": device_str
                }
            )
            logging.info("Wandb initialized")
        except Exception as e:
            logging.error(f"Failed to initialize wandb: {e}")
    
    logging.info(f"Learner initialized: {d} attributes -> {k} components")

@app.get("/get_params", response_model=ParametersResponse)
async def get_params():
    """Get current model parameters for behavior policy"""
    try:
        if model is None:
            return ParametersResponse(
                mask_logits=[], step=0, tau=0.0, 
                success=False, error="Model not initialized"
            )
        
        return ParametersResponse(
            mask_logits=model.mask_logits.detach().cpu().tolist(),
            step=current_step,
            tau=tau,
            success=True
        )
        
    except Exception as e:
        return ParametersResponse(
            mask_logits=[], step=0, tau=0.0,
            success=False, error=str(e)
        )

@app.post("/train_step", response_model=TrainStepResponse)
async def train_step_endpoint(request: TrainStepRequest):
    """Perform a single training step with provided batch data"""
    global current_step, tau
    
    try:
        if model is None or optimizer is None:
            return TrainStepResponse(
                success=False, step=current_step, loss=0.0, 
                reward_signal=0.0, active_attributes=0.0,
                error="Model not initialized"
            )
        
        # Convert data to tensors
        m_hard = torch.tensor(request.m_hard, device=device)
        R = torch.tensor(request.R, device=device)
        
        # Perform training step
        metrics = await train_step(m_hard, R, current_step)
        
        current_step += 1
        
        # Temperature annealing
        if current_step % 100 == 0 and current_step > 0:
            tau = max(0.1, tau * 0.995)
        
        # Periodic checkpointing (every 100 steps by default)
        if current_step % 100 == 0:
            save_checkpoint(current_step)
        
        # Wandb logging
        if wandb_run:
            active_features = torch.sigmoid(model.mask_logits).sum().item()
            wandb_run.log({
                **metrics,
                'step': current_step,
                'active_features': active_features,
                'temperature': tau,
            }, step=current_step)
        
        return TrainStepResponse(
            success=True,
            step=current_step,
            loss=metrics['loss'],
            reward_signal=metrics['reward_signal'],
            active_attributes=metrics['active_attributes']
        )
        
    except Exception as e:
        logging.error(f"Training step failed: {e}")
        return TrainStepResponse(
            success=False, step=current_step, loss=0.0,
            reward_signal=0.0, active_attributes=0.0,
            error=str(e)
        )

async def train_step(m_hard: torch.Tensor, R: torch.Tensor, step: int) -> Dict[str, float]:
    """Single training step with reward matrix using full model (like working test)"""
    optimizer.zero_grad()
    
    # R is now [batch_size, d] reward matrix - treat as input data
    x = R  # Use rewards as input data
    batch_size, d = R.shape
    
    # Normalize input like original gumbel.py
    x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2 normalize each sample
    
    # Forward pass - use the model's built-in forward method (with gradient flow)
    z, x_hat, masks = model.forward(x)
    
    # Reconstruction loss (exactly like original gumbel.py)
    recon_loss = torch.nn.functional.mse_loss(x_hat, x)
    
    # Sparsity loss - encourage masks to be sparse (exactly like original)
    mask_probs = torch.sigmoid(model.mask_logits)
    sparsity_loss = mask_probs.mean()  # Penalize high probabilities
    
    # Total loss (exactly like original gumbel.py)
    loss = recon_loss + model.sparsity_weight * sparsity_loss
    
    # Backward pass
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        'loss': float(loss.detach().cpu()),
        'reward_signal': float(recon_loss.detach().cpu()),  # Use recon_loss as reward signal
        'sparsity_loss': float(sparsity_loss.detach().cpu()),
        'avg_reward': float(R.mean().detach().cpu()) if R.numel() > 0 else 0.0,
        'active_attributes': float(masks.sum().detach().cpu()),  # Use model's masks
    }

async def save_checkpoint(step: int, final: bool = False):
    """Save model checkpoint"""
    try:
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step,
            'tau': tau,
        }
        
        filename = "final_checkpoint.pt" if final else f"checkpoint_step_{step}.pt"
        path = Path("./checkpoints") / filename
        
        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint: {path}")
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

@app.get("/parameters", response_model=ParametersResponse)
async def get_parameters():
    """Get current model parameters (for collector's behavior policy)"""
    try:
        if model is None:
            return ParametersResponse(
                mask_logits=[], step=0, tau=0.0, 
                success=False, error="Model not initialized"
            )
        
        return ParametersResponse(
            mask_logits=model.mask_logits.detach().cpu().tolist(),
            step=current_step,
            tau=tau,
            success=True
        )
        
    except Exception as e:
        return ParametersResponse(
            mask_logits=[], step=0, tau=0.0,
            success=False, error=str(e)
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get learner status"""
    active_features = 0.0
    if model is not None:
        active_features = torch.sigmoid(model.mask_logits).sum().item()
    
    return StatusResponse(
        status="running" if model is not None else "initializing",
        device=str(device) if device else "unknown",
        current_step=current_step,
        active_features=active_features
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": model is not None
    }

# Checkpointing endpoints
@app.post("/save_checkpoint")
async def save_checkpoint_endpoint():
    """Save current model checkpoint"""
    try:
        path = save_checkpoint(current_step, final=False)
        if path:
            return {"success": True, "checkpoint_path": path, "step": current_step}
        else:
            return {"success": False, "error": "Failed to save checkpoint"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/save_final_checkpoint")
async def save_final_checkpoint_endpoint():
    """Save final model checkpoint"""
    try:
        path = save_checkpoint(current_step, final=True)
        if path:
            return {"success": True, "checkpoint_path": path, "step": current_step}
        else:
            return {"success": False, "error": "Failed to save final checkpoint"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Save final checkpoint on shutdown"""
    logging.info("Shutting down learner server, saving final checkpoint...")
    save_checkpoint(current_step, final=True)
    
    if wandb_run:
        wandb_run.finish()
        logging.info("Finished wandb run")

def main():
    parser = argparse.ArgumentParser(description="Learner Server")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--k", type=int, default=10, help="Number of components")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.1, help="Sparsity weight")
    parser.add_argument("--tau-init", type=float, default=1.0, help="Initial temperature")
    
    # Communication no longer needed - coordinator handles orchestration
    
    # Server parameters
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for learner")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize learner on startup
    @app.on_event("startup")
    async def startup_event():
        initialize_learner(
            d=args.d,
            k=args.k,
            lr=args.lr,
            sparsity_weight=args.sparsity_weight,
            tau_init=args.tau_init,
            device_str=args.device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb
        )
    
    logging.info(f"Starting Learner Server on {args.host}:{args.port}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Model: {args.d} attributes -> {args.k} components")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()