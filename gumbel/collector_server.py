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
from data_sampler import create_data_sampler
from reward_scorer import VLLMRewardScorer
from utils import bernoulli_gumbel_soft
from vllm_server_standalone import startup_vllm_engine

# Request/Response models
class CollectionRequest(BaseModel):
    batch_size: int
    behavior_logits: List[float]
    tau: float

class CollectionResponse(BaseModel):
    X: List[List[float]]  # [batch_size, d]
    m_hard: List[float]   # [d]
    R: List[float]        # [batch_size]
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
reward_scorer = None
vllm_engine = None
collections_count = 0
device = None

async def initialize_collector(d: int, dataset_path: str, device_str: str, 
                             vllm_model: str, gpu_memory_util: float):
    """Initialize collector components"""
    global data_sampler, reward_scorer, vllm_engine, device
    
    device = torch.device(device_str)
    logging.info(f"Initializing collector on {device}")
    
    # Initialize VLLM engine for reward scoring
    logging.info("Starting VLLM engine for reward scoring...")
    try:
        await startup_vllm_engine(vllm_model, gpu_memory_util)
        from vllm_server_standalone import engine, sampling_params
        vllm_engine = engine
        logging.info("VLLM engine initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize VLLM engine: {e}")
        vllm_engine = None
    
    # Initialize data sampler
    logging.info("Initializing data sampler...")
    data_sampler = create_data_sampler({
        'dataset_path': dataset_path,
        'num_attributes': d,
        'feature_dim': d
    })
    
    # Initialize reward scorer
    logging.info("Initializing reward scorer...")
    attribute_prompts = [f"You are evaluating responses based on attribute {i}." for i in range(min(d, 10))]
    reward_scorer = VLLMRewardScorer(
        server_url="http://localhost:8000",  # Use local VLLM engine
        base_prompt="You are a helpful assistant.",
        attribute_prompts=attribute_prompts
    )
    
    stats = data_sampler.get_stats()
    logging.info(f"Collector initialized: {stats['num_users']} users, {stats['total_samples']} samples")

@app.post("/collect_batch", response_model=CollectionResponse)
async def collect_batch(request: CollectionRequest):
    """
    Main collection endpoint: sample data and compute rewards
    """
    global collections_count
    
    try:
        if data_sampler is None or reward_scorer is None:
            raise HTTPException(status_code=500, detail="Collector not initialized")
        
        batch_size = request.batch_size
        behavior_logits = torch.tensor(request.behavior_logits, device=device)
        tau = request.tau
        
        logging.debug(f"Collecting batch: size={batch_size}, tau={tau:.3f}")
        
        # Sample user data
        batch_sample = data_sampler.sample_batch(batch_size, device=device)
        X = batch_sample.X  # [batch_size, d]
        user_data = batch_sample.user_data
        
        # Sample hard mask using behavior policy
        with torch.no_grad():
            _, m_hard = bernoulli_gumbel_soft(behavior_logits, tau)  # [d]
        
        # Compute rewards using VLLM
        # Note: This uses the reward_scorer which will make HTTP requests
        # In a real setup, you'd use the local VLLM engine directly
        from reward_scorer import score_rewards
        R = score_rewards(X, m_hard, reward_scorer, user_data)  # [batch_size]
        
        collections_count += 1
        
        # Convert tensors to lists for JSON serialization
        response = CollectionResponse(
            X=X.detach().cpu().tolist(),
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
            X=[], m_hard=[], R=[], user_data={},
            success=False, error=str(e)
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get collector status"""
    return StatusResponse(
        status="running" if data_sampler is not None else "initializing",
        device=str(device) if device else "unknown",
        vllm_ready=vllm_engine is not None,
        collections_served=collections_count
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_sampler_ready": data_sampler is not None,
        "reward_scorer_ready": reward_scorer is not None,
        "vllm_ready": vllm_engine is not None
    }

class CollectorClient:
    """
    Client for communicating with collector server.
    Used by learner to request data collection.
    """
    
    def __init__(self, collector_url: str = "http://localhost:8001"):
        self.collector_url = collector_url.rstrip('/')
        self.session = requests.Session()
        
    def collect_batch(self, batch_size: int, behavior_logits: torch.Tensor, tau: float) -> Dict[str, Any]:
        """Request batch collection from collector server"""
        try:
            request_data = {
                "batch_size": batch_size,
                "behavior_logits": behavior_logits.tolist(),
                "tau": tau
            }
            
            response = self.session.post(
                f"{self.collector_url}/collect_batch",
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logging.error(f"Collector request failed: {response.status_code}")
                return None
            
            result = response.json()
            
            if not result["success"]:
                logging.error(f"Collector error: {result.get('error', 'Unknown error')}")
                return None
            
            # Convert back to tensors
            return {
                'X': torch.tensor(result['X'], dtype=torch.float32),
                'm_hard': torch.tensor(result['m_hard'], dtype=torch.float32),
                'R': torch.tensor(result['R'], dtype=torch.float32),
                'user_data': result['user_data']
            }
            
        except Exception as e:
            logging.error(f"Error communicating with collector: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get collector status"""
        try:
            response = self.session.get(f"{self.collector_url}/status", timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logging.error(f"Error getting collector status: {e}")
        return {"status": "unknown"}

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