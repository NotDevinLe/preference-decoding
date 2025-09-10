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
import requests

# Local imports
from skeleton import SparseMaskModel
from collector_server import CollectorClient

# Wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Request/Response models
class TrainingRequest(BaseModel):
    max_steps: int = 1000
    log_freq: int = 20
    checkpoint_freq: int = 100

class TrainingResponse(BaseModel):
    success: bool
    final_step: int
    final_active_features: float
    final_sparsity_ratio: float
    error: str = None

class ParametersRequest(BaseModel):
    pass

class ParametersResponse(BaseModel):
    mask_logits: List[float]
    step: int
    tau: float
    success: bool
    error: str = None

class StatusResponse(BaseModel):
    status: str
    device: str
    current_step: int
    active_features: float
    training_active: bool

# Global variables
app = FastAPI(title="Learner Server", version="1.0")
model = None
optimizer = None
collector_client = None
device = None
current_step = 0
tau = 1.0
training_active = False
wandb_run = None

def initialize_learner(d: int, k: int, lr: float, sparsity_weight: float,
                      tau_init: float, device_str: str, collector_url: str,
                      checkpoint_dir: str, use_wandb: bool):
    """Initialize learner components"""
    global model, optimizer, collector_client, device, tau, wandb_run
    
    device = torch.device(device_str)
    tau = tau_init
    
    logging.info(f"Initializing learner on {device}")
    
    # Initialize model
    model = SparseMaskModel(d, k, sparsity_weight=sparsity_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize collector client
    collector_client = CollectorClient(collector_url)
    
    # Test collector connection
    status = collector_client.get_status()
    logging.info(f"Collector status: {status}")
    
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

@app.post("/start_training", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """Start the training loop"""
    global training_active, current_step, tau
    
    if training_active:
        return TrainingResponse(
            success=False, 
            final_step=current_step,
            final_active_features=0.0,
            final_sparsity_ratio=0.0,
            error="Training already active"
        )
    
    if model is None or collector_client is None:
        return TrainingResponse(
            success=False,
            final_step=0,
            final_active_features=0.0, 
            final_sparsity_ratio=0.0,
            error="Learner not initialized"
        )
    
    training_active = True
    
    try:
        logging.info(f"Starting training for {request.max_steps} steps")
        
        for step in range(request.max_steps):
            current_step = step
            
            # Request batch from collector
            batch_data = collector_client.collect_batch(
                batch_size=32,  # Default batch size
                behavior_logits=model.mask_logits.detach().cpu(),
                tau=tau
            )
            
            if batch_data is None:
                logging.error(f"Failed to get batch from collector at step {step}")
                continue
            
            # Move data to device
            X = batch_data['X'].to(device)
            m_hard = batch_data['m_hard'].to(device) 
            R = batch_data['R'].to(device)
            
            # Training step
            metrics = await train_step(X, m_hard, R, step)
            
            # Logging
            if step % request.log_freq == 0:
                active_features = torch.sigmoid(model.mask_logits).sum().item()
                
                logging.info(
                    f"Step {step}: loss={metrics['loss']:.4f} "
                    f"recon={metrics['reconstruction_loss']:.4f} "
                    f"sparsity={metrics['sparsity_loss']:.4f} "
                    f"reward={metrics['avg_reward']:.4f} "
                    f"active={active_features:.1f} "
                    f"tau={tau:.3f}"
                )
                
                # Wandb logging
                if wandb_run:
                    wandb_run.log({
                        **metrics,
                        'step': step,
                        'active_features': active_features,
                        'temperature': tau,
                    }, step=step)
            
            # Temperature annealing
            if step % 100 == 0 and step > 0:
                tau = max(0.1, tau * 0.995)
            
            # Checkpointing
            if step % request.checkpoint_freq == 0 and step > 0:
                await save_checkpoint(step)
        
        # Final results
        final_active = torch.sigmoid(model.mask_logits).sum().item()
        final_sparsity = final_active / model.mask_logits.shape[0]
        
        await save_checkpoint(current_step, final=True)
        
        logging.info(f"Training completed after {current_step + 1} steps")
        logging.info(f"Final sparse mask: {final_active:.1f} active ({100*final_sparsity:.1f}%)")
        
        training_active = False
        
        return TrainingResponse(
            success=True,
            final_step=current_step,
            final_active_features=final_active,
            final_sparsity_ratio=final_sparsity
        )
        
    except Exception as e:
        training_active = False
        logging.error(f"Training failed: {e}")
        return TrainingResponse(
            success=False,
            final_step=current_step,
            final_active_features=0.0,
            final_sparsity_ratio=0.0,
            error=str(e)
        )

async def train_step(X: torch.Tensor, m_hard: torch.Tensor, R: torch.Tensor, step: int) -> Dict[str, float]:
    """Single training step"""
    optimizer.zero_grad()
    
    # Forward pass
    mask_logits = model.mask_logits
    xhat, _, _ = model.forward_decode_hard_soft(X, mask_logits, tau, gated_st=True)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(xhat, X, reduction="mean")
    
    # Sparsity loss (on active attributes from behavior policy)
    idx_on = torch.nonzero(m_hard, as_tuple=False).squeeze(1)
    if idx_on.numel() > 0:
        sparsity_loss = torch.sigmoid(mask_logits.index_select(0, idx_on)).mean()
    else:
        sparsity_loss = torch.tensor(0.0, device=device)
    
    # Total loss (reward-weighted)
    task_loss = recon_loss + model.sparsity_weight * sparsity_loss
    reward_weight = R.abs().mean() if R.numel() > 0 else torch.tensor(1.0, device=device)
    loss = reward_weight * task_loss
    
    # Backward pass
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        'loss': float(loss.detach().cpu()),
        'reconstruction_loss': float(recon_loss.detach().cpu()),
        'sparsity_loss': float(sparsity_loss.detach().cpu()),
        'reward_weight': float(reward_weight.detach().cpu()),
        'avg_reward': float(R.mean().detach().cpu()) if R.numel() > 0 else 0.0,
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
        active_features=active_features,
        training_active=training_active
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "collector_connected": collector_client is not None,
        "training_active": training_active
    }

@app.post("/stop_training")
async def stop_training():
    """Stop training (emergency stop)"""
    global training_active
    training_active = False
    return {"success": True, "message": "Training stopped"}

def main():
    parser = argparse.ArgumentParser(description="Learner Server")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--k", type=int, default=10, help="Number of components")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.1, help="Sparsity weight")
    parser.add_argument("--tau-init", type=float, default=1.0, help="Initial temperature")
    
    # Communication
    parser.add_argument("--collector-url", type=str, default="http://localhost:8001", 
                       help="Collector server URL")
    
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
            collector_url=args.collector_url,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb
        )
    
    logging.info(f"Starting Learner Server on {args.host}:{args.port}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Collector URL: {args.collector_url}")
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