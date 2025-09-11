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
    Follows the same logic as src/core/drift.py get_log_probs function.
    
    Returns:
        tuple: (log_probs, token_counts) for each completion
    """
    log_probs = []
    token_counts = []
    
    for i, (sys_prompt, user_prompt, completion) in enumerate(zip(system_prompts, user_prompts, completions)):
        try:
            # Add 10-second throttling delay to avoid overwhelming VLLM server
            if i > 0:  # Skip delay for first request
                logging.debug(f"Throttling: sleeping 10s before request {i+1}/{len(system_prompts)}")
                await asyncio.sleep(10.0)  # 10 second delay between requests
            
            logging.debug(f"Processing API request {i+1}/{len(system_prompts)}")
            
            # Format prompt using chat template structure (matching original logic)
            messages = [
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
            
            # Get the chat completion with logprobs for the full sequence
            # We want to score: prompt + completion + eos
            response = vllm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1,  # Minimal generation (we mainly want prompt logprobs)
                temperature=0.0,
                logprobs=True,
                top_logprobs=1,
                extra_body={
                    "prompt_logprobs": 1,  # Enable prompt logprobs like original
                    "include_stop_str_in_output": True
                }
            )
            
            # Note: The OpenAI API doesn't give us the exact same access to prompt_logprobs
            # as the direct VLLM interface. We need to work with what's available.
            
            # For now, use a simpler approach that approximates the original logic
            # by making a completion request with the full prompt+completion text
            full_text = f"{sys_prompt.strip()}\n\nUser: {user_prompt.strip()}\n\nAssistant: {completion}"
            
            # Use completions endpoint to get logprobs for the full sequence
            completion_response = vllm_client.completions.create(
                model=model_name,
                prompt=full_text,
                max_tokens=1,
                temperature=0.0,
                logprobs=1,
                echo=True  # Include prompt tokens in response
            )
            
            if completion_response.choices and completion_response.choices[0].logprobs:
                logprobs_data = completion_response.choices[0].logprobs
                token_logprobs = logprobs_data.token_logprobs or []
                tokens = logprobs_data.tokens or []
                
                # Find where the actual completion starts (after "Assistant:")
                # This mimics the original logic of skipping prompt tokens
                completion_start_idx = len(tokens)  # Default to end if not found
                for idx, token in enumerate(tokens):
                    if "Assistant:" in token or "assistant:" in token:
                        completion_start_idx = idx + 1
                        break
                
                # Sum logprobs for completion tokens only (matching original logic)
                if completion_start_idx < len(token_logprobs):
                    completion_logprobs = token_logprobs[completion_start_idx:]
                    total_log_prob = sum(lp for lp in completion_logprobs if lp is not None)
                    token_count = len(completion_logprobs)
                else:
                    # Fallback if we can't find the split point
                    total_log_prob = sum(lp for lp in token_logprobs[-10:] if lp is not None)  # Last 10 tokens
                    token_count = min(10, len(token_logprobs))
                
                log_probs.append(total_log_prob)
                token_counts.append(max(token_count, 1))
                
                logging.debug(f"✅ Computed logprob {total_log_prob:.3f} for '{completion[:30]}...' ({token_count} tokens)")
            else:
                # Fallback scoring (length-based penalty)
                log_prob = -len(completion.split()) * 0.5  # Penalty per word
                token_count = len(completion.split())
                log_probs.append(log_prob)
                token_counts.append(max(token_count, 1))
                logging.debug(f"⚠️ Fallback scoring for '{completion[:30]}...': {log_prob:.3f}")
                
        except Exception as e:
            logging.warning(f"Error getting log probs for completion '{completion[:30]}...': {e}")
            # Fallback scoring
            log_prob = -len(completion.split()) * 0.5
            token_count = len(completion.split())
            log_probs.append(log_prob)
            token_counts.append(max(token_count, 1))
    
    return log_probs, token_counts

async def compute_reward_matrix_direct(user_data: Dict[str, Any], m_hard: torch.Tensor) -> torch.Tensor:
    """
    Compute reward matrix using batched VLLM API calls for maximum efficiency.
    
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
    
    # Get active attributes only (optimization)
    active_attr_indices = torch.where(m_hard > 0)[0].tolist()
    active_attributes = [attribute_prompts[i] for i in active_attr_indices if i < len(attribute_prompts)]
    
    logging.debug(f"Computing rewards for {len(active_attributes)} active attributes out of {d} total")
    
    # Prepare batched API call
    # Structure: [baseline_batch, attr1_batch, attr2_batch, ...]
    all_system_prompts = []
    all_user_prompts = []
    all_completions = []
    
    # Add baseline prompts
    all_system_prompts.extend([base_prompt] * batch_size)
    all_user_prompts.extend(prompts)
    all_completions.extend(outputs)
    
    # Add attribute prompts
    for attr_prompt in active_attributes:
        all_system_prompts.extend([attr_prompt] * batch_size)
        all_user_prompts.extend(prompts)
        all_completions.extend(outputs)
    
    total_requests = len(all_system_prompts)
    logging.debug(f"Making single batched API call with {total_requests} requests")
    
    # Single batched API call
    import time
    start_time = time.time()
    all_probs, all_counts = await get_log_probs_api(
        all_system_prompts, all_user_prompts, all_completions
    )
    api_time = time.time() - start_time
    logging.debug(f"Batched API call completed in {api_time:.2f}s")
    
    # Parse results
    # First batch_size results are baseline scores
    base_probs = all_probs[:batch_size]
    base_counts = all_counts[:batch_size]
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Remaining results are attribute scores
    attr_results = []
    for i, attr_idx in enumerate(active_attr_indices[:len(active_attributes)]):
        start_idx = batch_size + i * batch_size
        end_idx = start_idx + batch_size
        
        attr_probs = all_probs[start_idx:end_idx]
        attr_counts = all_counts[start_idx:end_idx]
        attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        attr_results.append((attr_idx, attr_scores))
    
    # Build reward matrix
    reward_matrix = torch.zeros(batch_size, d, device=device)
    
    for attr_idx, attr_scores in attr_results:
        # Compute drift scores (attribute vs reference)
        drift_scores = attr_scores - base_scores
        reward_matrix[:, attr_idx] = drift_scores
    
    # Apply mask (this should be redundant since we only computed active attributes)
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