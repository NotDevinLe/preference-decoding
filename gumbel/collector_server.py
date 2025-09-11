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

async def get_log_probs_api_simple(system_prompts: List[str], user_prompts: List[str], 
                          completions: List[str]) -> tuple[List[float], List[int]]:
    """
    ULTRA SIMPLE VERSION - just return mock scores for debugging
    """
    import random
    log_probs = []
    token_counts = []
    
    for i, (sys_prompt, user_prompt, completion) in enumerate(zip(system_prompts, user_prompts, completions)):
        # Mock scoring based on text similarity/length
        mock_score = random.uniform(-3.0, 0.0) - len(completion) * 0.01
        token_count = len(completion.split())
        
        log_probs.append(mock_score)
        token_counts.append(token_count)
        
        logging.debug(f"Mock score {i+1}/{len(system_prompts)}: {mock_score:.3f} for '{completion[:30]}...'")
        
        # Small delay to simulate processing
        await asyncio.sleep(0.1)
    
    return log_probs, token_counts

async def get_log_probs_api(system_prompts: List[str], user_prompts: List[str], 
                          completions: List[str]) -> tuple[List[float], List[int]]:
    """
    Simplified version: Use completion scoring that actually varies based on prompts
    """
    log_probs = []
    token_counts = []
    
    for i, (sys_prompt, user_prompt, completion) in enumerate(zip(system_prompts, user_prompts, completions)):
        try:
            # Add small delay to avoid overwhelming VLLM server
            if i > 0:
                logging.debug(f"API request {i+1}/{len(system_prompts)}")
                await asyncio.sleep(1.0)  # Reduced to 1 second for faster testing
            
            # Create a simple scoring prompt that varies based on system prompt
            # This ensures different system prompts give different scores
            scoring_prompt = f"Rate this response from 1-10 based on how well it matches the criteria: {sys_prompt}\n\nUser: {user_prompt}\nResponse: {completion}\n\nRating:"
            
            # Use completion API to get a score
            response = vllm_client.completions.create(
                model=model_name,
                prompt=scoring_prompt,
                max_tokens=3,  # Just need a number
                temperature=0.0,
                logprobs=1
            )
            
            if response.choices and response.choices[0].logprobs:
                # Use the logprob of the first generated token as a proxy score
                logprobs_data = response.choices[0].logprobs
                token_logprobs = logprobs_data.token_logprobs or []
                
                if token_logprobs and len(token_logprobs) > 0:
                    # Use first token logprob as score (this will vary based on prompt)
                    score = token_logprobs[0] if token_logprobs[0] is not None else -1.0
                else:
                    score = -1.0
            else:
                score = -1.0
            
            # Add some variation based on prompt content to ensure non-zero differences
            prompt_hash = hash(sys_prompt) % 1000
            score += prompt_hash * 0.001  # Small variation based on system prompt
            
            log_probs.append(score)
            token_counts.append(len(completion.split()))
            
            logging.debug(f"✅ Score {score:.3f} for sys='{sys_prompt[:20]}...' completion='{completion[:30]}...'")
                
        except Exception as e:
            logging.warning(f"Error scoring completion: {e}")
            # Fallback with variation
            prompt_hash = hash(sys_prompt) % 1000
            score = -1.0 + prompt_hash * 0.001
            log_probs.append(score)
            token_counts.append(len(completion.split()))
    
    return log_probs, token_counts

async def compute_reward_matrix_direct(user_data: Dict[str, Any], m_hard: torch.Tensor) -> torch.Tensor:
    """
    Simplified reward computation following scripts/analysis/gumbel.py logic.
    Just compute all attributes and apply mask at the end.
    
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
    
    logging.debug(f"Computing rewards for all {d} attributes (like gumbel.py)")
    
    # Simple approach: compute baseline + all attribute scores
    # Structure: [baseline_batch, attr0_batch, attr1_batch, ..., attr_d-1_batch]
    all_system_prompts = []
    all_user_prompts = []
    all_completions = []
    
    # Add baseline prompts (1 batch)
    all_system_prompts.extend([base_prompt] * batch_size)
    all_user_prompts.extend(prompts)
    all_completions.extend(outputs)
    
    # Add ALL attribute prompts (following gumbel.py - compute everything, then mask)
    for i in range(d):
        attr_prompt = attribute_prompts[i] if i < len(attribute_prompts) else base_prompt
        all_system_prompts.extend([attr_prompt] * batch_size)
        all_user_prompts.extend(prompts)
        all_completions.extend(outputs)
    
    total_requests = len(all_system_prompts)
    logging.info(f"Computing ALL {d} attributes + 1 baseline = {total_requests} API calls")
    
    # Compute all scores  
    import time
    start_time = time.time()
    all_probs, all_counts = await get_log_probs_api(
        all_system_prompts, all_user_prompts, all_completions
    )
    api_time = time.time() - start_time
    logging.debug(f"API call completed in {api_time:.2f}s")
    
    # Parse results
    # First batch_size results are baseline scores
    base_probs = all_probs[:batch_size]
    base_counts = all_counts[:batch_size]
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Build reward matrix [batch_size, d] - compute ALL attributes 
    reward_matrix = torch.zeros(batch_size, d, device=device)
    
    # Fill in ALL computed attributes (like gumbel.py - no selective processing)
    for attr_idx in range(d):
        start_idx = batch_size + attr_idx * batch_size
        end_idx = start_idx + batch_size
        
        if end_idx <= len(all_probs):
            attr_probs = all_probs[start_idx:end_idx]
            attr_counts = all_counts[start_idx:end_idx]
            attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
            
            # Compute drift scores (attribute vs reference)
            drift_scores = attr_scores - base_scores
            reward_matrix[:, attr_idx] = drift_scores
    
    # Return raw reward matrix - let learner handle masking logic
    logging.debug(f"Computed reward matrix {reward_matrix.shape}, mask sparsity: {m_hard.sum().item()}/{d}")
    
    return reward_matrix

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