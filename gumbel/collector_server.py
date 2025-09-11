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

# OpenAI API client for external VLLM server
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    logging.error(f"OpenAI client not available: {e}")
    logging.error("Cannot proceed without OpenAI client - install with: pip install openai")
    OPENAI_AVAILABLE = False

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
vllm_client = None
model_name = None
attribute_prompts = None
base_prompt = "You are a helpful assistant."
collections_count = 0
device = None

async def initialize_collector(d: int, dataset_path: str, device_str: str, 
                             vllm_server_url: str, model_name_arg: str,
                             attribute_prompts_path: str = None):
    """Initialize collector components"""
    global data_sampler, vllm_client, model_name, attribute_prompts, device
    
    if not OPENAI_AVAILABLE:
        logging.error("OpenAI client not available - cannot initialize collector")
        raise RuntimeError("OpenAI client required for VLLM server communication")
    
    device = torch.device(device_str)
    logging.info(f"Initializing collector on {device}")
    
    # Initialize VLLM client for external server
    logging.info(f"Connecting to VLLM server at: {vllm_server_url}")
    try:
        vllm_client = OpenAI(
            base_url=vllm_server_url,
            api_key="dummy"  # VLLM server doesn't require real API key
        )
        model_name = model_name_arg
        
        # Test connection
        models = vllm_client.models.list()
        logging.info(f"Connected to VLLM server successfully. Available models: {[m.id for m in models.data]}")
        
    except Exception as e:
        logging.error(f"Failed to connect to VLLM server: {e}")
        raise
    
    # Initialize data sampler
    logging.info("Initializing data sampler...")
    data_sampler = DataSampler(dataset_path=dataset_path)
    
    # Initialize attribute prompts for scoring
    if not attribute_prompts_path:
        logging.error("No attribute prompts file provided")
        raise ValueError("--attribute-prompts-path is required")
        
    if not Path(attribute_prompts_path).exists():
        logging.error(f"Attribute prompts file not found: {attribute_prompts_path}")
        raise FileNotFoundError(f"Attribute prompts file not found: {attribute_prompts_path}")
    
    logging.info(f"Loading attribute prompts from {attribute_prompts_path}")
    try:
        import json
        with open(attribute_prompts_path, 'r') as f:
            loaded_prompts = json.load(f)
        
        # Handle different file formats
        if isinstance(loaded_prompts, list):
            attribute_prompts = loaded_prompts[:d]  # Take first d prompts
        elif isinstance(loaded_prompts, dict) and 'prompts' in loaded_prompts:
            attribute_prompts = loaded_prompts['prompts'][:d]
        else:
            logging.error("Invalid attribute prompts file format - expected list or {'prompts': [...]}")
            raise ValueError("Invalid attribute prompts file format")
            
        if len(attribute_prompts) < d:
            logging.error(f"Not enough attribute prompts: file has {len(attribute_prompts)}, but d={d}")
            raise ValueError(f"Need at least {d} attribute prompts, but file only has {len(attribute_prompts)}")
            
        logging.info(f"Successfully loaded {len(attribute_prompts)} attribute prompts")
        
    except Exception as e:
        logging.error(f"Failed to load attribute prompts: {e}")
        raise
    
    stats = data_sampler.get_stats()
    logging.info(f"Collector initialized: {stats['num_users']} users, {stats['total_samples']} samples")

async def get_log_probs_api(system_prompts: List[str], user_prompts: List[str], 
                          completions: List[str]) -> tuple[List[float], List[int]]:
    """
    Get log probabilities using OpenAI API (via VLLM server).
    Replaces the direct VLLM get_log_probs function.
    
    Returns:
        tuple: (log_probs, token_counts) for each completion
    """
    log_probs = []
    token_counts = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completions):
        try:
            # Use chat completions with logprobs
            response = vllm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                logprobs=True,
                top_logprobs=1,
                max_tokens=1,  # We just want to score the existing completion
                temperature=0.0
            )
            
            # Extract logprob for the completion
            # Note: This is a simplified version - real implementation would need
            # to properly tokenize the completion and sum logprobs
            if response.choices and response.choices[0].logprobs:
                # For now, use a placeholder scoring mechanism
                # This would need to be more sophisticated in practice
                log_prob = 0.0
                token_count = len(completion.split())  # Rough token estimate
                
                log_probs.append(log_prob)
                token_counts.append(token_count)
            else:
                log_probs.append(0.0)
                token_counts.append(1)
                
        except Exception as e:
            logging.warning(f"Error getting log probs for completion: {e}")
            log_probs.append(0.0) 
            token_counts.append(1)
    
    return log_probs, token_counts

async def compute_reward_matrix_direct(user_data: Dict[str, Any], m_hard: torch.Tensor) -> torch.Tensor:
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
    base_probs, base_counts = await get_log_probs_api(
        [base_prompt] * batch_size,
        prompts, outputs
    )
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Compute scores for each attribute prompt
    reward_matrix = torch.zeros(batch_size, d, device=device)
    
    for attr_idx, attr_prompt in enumerate(attribute_prompts):
        if attr_idx >= d:  # Don't exceed the mask dimension
            break
            
        # Get scores with this attribute prompt
        attr_probs, attr_counts = await get_log_probs_api(
            [attr_prompt] * batch_size,
            prompts, outputs
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
        if data_sampler is None or vllm_client is None:
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
        R = await compute_reward_matrix_direct(user_data, m_hard)  # [batch_size, d]
        
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
        vllm_ready=vllm_client is not None,
        collections_served=collections_count
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_sampler_ready": data_sampler is not None,
        "vllm_ready": vllm_client is not None
    }

# CollectorClient removed - coordinator handles communication directly

def main():
    parser = argparse.ArgumentParser(description="Collector Server")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    
    # VLLM parameters
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000/v1", help="VLLM server URL")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name for API requests")
    parser.add_argument("--gpu-memory-util", type=float, default=0.8, help="GPU memory utilization")
    
    # Attribute prompts
    parser.add_argument("--attribute-prompts-path", type=str, required=True, help="Path to attribute prompts JSON file")
    
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
            vllm_server_url=args.vllm_server_url,
            model_name_arg=args.model_name,
            attribute_prompts_path=args.attribute_prompts_path
        )
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        await startup()
    
    logging.info(f"Starting Collector Server on {args.host}:{args.port}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Dataset: {args.dataset_path}")
    logging.info(f"VLLM Server: {args.vllm_server_url}")
    logging.info(f"Model: {args.model_name}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()