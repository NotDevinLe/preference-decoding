#!/usr/bin/env python3
"""
Collector Server: Handles data sampling and reward scoring using VLLM.
Runs on GPU 0, communicates with learner server via HTTP.
"""

import asyncio
import json
import logging
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests

# Local imports
from sampler import DataSampler
from utils import bernoulli_gumbel_soft

# VLLM imports
try:
    from vllm import LLM
    from transformers import AutoTokenizer
    import sys
    from pathlib import Path
    # Add project root to path for core imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.core.drift import get_log_probs
    VLLM_AVAILABLE = True
except ImportError as e:
    logging.error(f"VLLM dependencies not available: {e}")
    logging.error("Cannot proceed without VLLM - exiting")
    VLLM_AVAILABLE = False

# Request/Response models
class CollectionRequest(BaseModel):
    users_per_batch: int
    samples_per_user: int
    behavior_logits: List[float]
    tau: float

class CollectionResponse(BaseModel):
    m_hard: List[float]   # [d]
    R: List[List[float]]  # [batch_size, d] - reward matrix
    user_data: Dict[str, Any]
    success: bool
    error: str = None

class StatusResponse(BaseModel):
    status: str
    device: str
    vllm_ready: bool
    collections_served: int

# Global variables
app = FastAPI(title="Collector Server", version="1.0")
data_sampler = None
vllm_model = None
vllm_tokenizer = None
attribute_prompts = None
base_prompt = "You are a helpful assistant."
collections_count = 0
device = None

async def initialize_collector(d: int, dataset_path: str, device_str: str, 
                             vllm_model_name: str, gpu_memory_util: float):
    """Initialize collector components"""
    global data_sampler, vllm_model, vllm_tokenizer, attribute_prompts, device
    
    if not VLLM_AVAILABLE:
        logging.error("VLLM not available - cannot initialize collector")
        raise RuntimeError("VLLM dependencies required")
    
    device = torch.device(device_str)
    logging.info(f"Initializing collector on {device}")
    
    # Initialize VLLM model directly
    logging.info(f"Loading VLLM model: {vllm_model_name}")
    try:
        vllm_model = LLM(
            model=vllm_model_name,
            tensor_parallel_size=1,
            dtype="bfloat16" if device_str == "cuda" else "float32",
            gpu_memory_utilization=gpu_memory_util
        )
        vllm_tokenizer = AutoTokenizer.from_pretrained(vllm_model_name)
        logging.info("VLLM model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load VLLM model: {e}")
        raise
    
    # Initialize data sampler
    logging.info("Initializing data sampler...")
    data_sampler = DataSampler(dataset_path=dataset_path)
    
    # Initialize attribute prompts for scoring
    attribute_prompts = [f"You are evaluating responses based on attribute {i}." for i in range(min(d, 10))]
    logging.info(f"Using {len(attribute_prompts)} attribute prompts for scoring")
    
    stats = data_sampler.get_stats()
    logging.info(f"Collector initialized: {stats['num_users']} users, {stats['total_samples']} samples")

def compute_reward_matrix_direct(user_data: Dict[str, Any], m_hard: torch.Tensor) -> torch.Tensor:
    """
    Compute reward matrix using direct VLLM scoring (drift-based like compute_reward_matrix_flexible_efficient.py)
    
    Args:
        user_data: Dict with 'prompts', 'outputs', 'user_ids'
        m_hard: [d] binary mask for active attributes
        
    Returns:
        torch.Tensor: [batch_size, d] reward matrix
    """
    prompts = user_data['prompts']
    outputs = user_data['outputs']
    batch_size = len(prompts)
    d = len(m_hard)
    
    if batch_size == 0:
        return torch.zeros(0, d, device=device)
    
    # Compute baseline scores with reference prompt
    base_probs, base_counts = get_log_probs(
        vllm_model, vllm_tokenizer,
        [base_prompt] * batch_size,
        prompts, outputs, device
    )
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Compute scores for each attribute prompt
    reward_matrix = torch.zeros(batch_size, d, device=device)
    
    for attr_idx, attr_prompt in enumerate(attribute_prompts):
        if attr_idx >= d:  # Don't exceed the mask dimension
            break
            
        # Get scores with this attribute prompt
        attr_probs, attr_counts = get_log_probs(
            vllm_model, vllm_tokenizer,
            [attr_prompt] * batch_size,
            prompts, outputs, device
        )
        attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Compute drift scores (attribute vs reference)
        reward_matrix[:, attr_idx] = attr_scores - base_scores
    
    # Apply mask - zero out inactive attributes
    masked_reward_matrix = reward_matrix * m_hard.unsqueeze(0)
    
    logging.debug(f"Computed reward matrix {masked_reward_matrix.shape}, mask sparsity: {m_hard.sum().item()}/{d}")
    
    return masked_reward_matrix

@app.post("/generate_batch", response_model=CollectionResponse)
async def generate_batch(request: CollectionRequest):
    """
    Generate batch endpoint: sample data and compute rewards
    """
    global collections_count
    
    try:
        if data_sampler is None or vllm_model is None:
            raise HTTPException(status_code=500, detail="Collector not initialized")
        
        users_per_batch = request.users_per_batch
        samples_per_user = request.samples_per_user
        behavior_logits = torch.tensor(request.behavior_logits, device=device)
        tau = request.tau
        
        logging.debug(f"Collecting batch: users={users_per_batch}, samples_per_user={samples_per_user}, tau={tau:.3f}")
        
        # Sample user data
        user_data = data_sampler(users_per_batch=users_per_batch, samples_per_user=samples_per_user, device=device)
        
        # Sample hard mask using behavior policy
        with torch.no_grad():
            _, m_hard = bernoulli_gumbel_soft(behavior_logits, tau)  # [d]
        
        # Compute reward matrix using direct VLLM
        R = compute_reward_matrix_direct(user_data, m_hard)  # [batch_size, d]
        
        collections_count += 1
        
        # Convert tensors to lists for JSON serialization
        response = CollectionResponse(
            m_hard=m_hard.detach().cpu().tolist(), 
            R=R.detach().cpu().tolist(),
            user_data=user_data,
            success=True
        )
        
        logging.debug(f"Collection complete: reward_range=[{R.min():.3f}, {R.max():.3f}], mask_sparsity={m_hard.sum():.0f}")
        
        return response
        
    except Exception as e:
        logging.error(f"Error in collect_batch: {e}")
        return CollectionResponse(
            m_hard=[], R=[], user_data={},
            success=False, error=str(e)
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get collector status"""
    return StatusResponse(
        status="running" if data_sampler is not None else "initializing",
        device=str(device) if device else "unknown",
        vllm_ready=vllm_model is not None,
        collections_served=collections_count
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_sampler_ready": data_sampler is not None,
        "vllm_ready": vllm_model is not None
    }

# CollectorClient removed - coordinator handles communication directly

def main():
    parser = argparse.ArgumentParser(description="Collector Server")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    
    # VLLM parameters
    parser.add_argument("--vllm-model", type=str, default="microsoft/DialoGPT-medium", help="VLLM model")
    parser.add_argument("--gpu-memory-util", type=float, default=0.8, help="GPU memory utilization")
    
    # Server parameters
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for collector")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector on startup
    async def startup():
        await initialize_collector(
            d=args.d,
            dataset_path=args.dataset_path,
            device_str=args.device,
            vllm_model=args.vllm_model,
            gpu_memory_util=args.gpu_memory_util
        )
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        await startup()
    
    logging.info(f"Starting Collector Server on {args.host}:{args.port}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Dataset: {args.dataset_path}")
    logging.info(f"VLLM Model: {args.vllm_model}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()