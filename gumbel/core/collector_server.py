#!/usr/bin/env python3
"""
Collector Server: Handles data sampling and reward scoring using VLLM.
Runs on GPU 0, communicates with learner server via HTTP.
"""

import asyncio
import json
import logging
import argparse
from typing import List, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Local imports
from sampler import DataSampler
import aiohttp


# Request/Response models
class CollectionRequest(BaseModel):
    users_per_batch: int
    samples_per_user: int

class CollectionResponse(BaseModel):
    R: List[List[float]]  # [batch_size, d] - reward matrix
    user_data: Dict[str, Any]
    success: bool
    error: str = None

class StatusResponse(BaseModel):
    status: str
    collections_served: int

# Global variables
app = FastAPI()
data_sampler = None
attribute_prompts = None
base_prompt = "You are a helpful assistant."
collections_count = 0
device = None
vllm_server_url = None
model_name = None

def initialize_collector(d: int, dataset_path: str, device_str: str, attribute_prompts_path: str, vllm_server_url_arg: str, model_name_arg: str):
    """Initialize collector components"""
    global data_sampler, attribute_prompts, device, vllm_server_url, model_name
    
    device = torch.device(device_str)
    data_sampler = DataSampler(dataset_path=dataset_path)
    
    # Store VLLM server URL for aiohttp requests
    vllm_server_url = vllm_server_url_arg
    model_name = model_name_arg
    
    with open(attribute_prompts_path, 'r') as f:
        loaded_prompts = json.load(f)
    
    if isinstance(loaded_prompts, list):
        attribute_prompts = loaded_prompts[:d]
    elif isinstance(loaded_prompts, dict) and 'prompts' in loaded_prompts:
        attribute_prompts = loaded_prompts['prompts'][:d]
    else:
        raise ValueError("Invalid attribute prompts file format")
        
    if len(attribute_prompts) < d:
        raise ValueError(f"Need at least {d} attribute prompts")


async def get_log_probs_from_server(system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], temperature: float = 0.0) -> tuple[List[float], List[int]]:
    """Batch version using concurrent aiohttp calls to VLLM server"""
    
    async def single_request(session, sys_prompt, user_prompt, completion):
        try:
            messages = [
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
            
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": len(completion.split()) + 10,
                "temperature": temperature,
                "logprobs": True
            }
            
            async with session.post(f"{vllm_server_url}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    return -1.0, len(completion.split())
                
                result = await response.json()
                
                if result.get("choices") and result["choices"][0].get("logprobs"):
                    choice = result["choices"][0]
                    if choice["logprobs"] and choice["logprobs"].get("content"):
                        # Sum logprobs from completion tokens
                        completion_logprob = sum(
                            token_logprob["logprob"] 
                            for token_logprob in choice["logprobs"]["content"]
                            if token_logprob.get("logprob") is not None
                        )
                        return completion_logprob, len(choice["logprobs"]["content"])
                    else:
                        return -1.0, len(completion.split())
                else:
                    return -1.0, len(completion.split())
                
        except Exception:
            return -1.0, len(completion.split())
    
    # Create aiohttp session and run all requests concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            single_request(session, sys_prompt, user_prompt, completion)
            for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts)
        ]
        
        # Execute all requests in parallel
        results = await asyncio.gather(*tasks)
    
    # Separate log_probs and token_counts
    log_probs = [result[0] for result in results]
    token_counts = [result[1] for result in results]
    
    return log_probs, token_counts


async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """Compute reward matrix using VLLM server drift scoring"""
    prompts = user_data['prompts']
    outputs = user_data['outputs']
    batch_size = len(outputs)
    
    if batch_size == 0:
        return torch.zeros(0, d, device=device)
    
    # Compute base log probabilities using VLLM server
    base_probs, base_counts = await get_log_probs_from_server(
        [base_prompt] * batch_size, prompts, outputs
    )
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Build reward matrix
    reward_matrix = torch.zeros(batch_size, d, device=device)
    
    # Compute drift scores for each attribute
    for attr_idx in range(d):
        attr_prompt = attribute_prompts[attr_idx]
        
        # Get log probabilities for this attribute using VLLM server
        attr_probs, attr_counts = await get_log_probs_from_server(
            [attr_prompt] * batch_size, prompts, outputs
        )
        attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Drift score = attribute_score - base_score (from drift.py formula)
        reward_matrix[:, attr_idx] = attr_scores - base_scores
    
    return reward_matrix

@app.post("/generate_batch", response_model=CollectionResponse)
async def generate_batch(request: CollectionRequest):
    """
    Generate batch endpoint: sample data and compute rewards
    """
    global collections_count
    
    try:
        if data_sampler is None:
            raise HTTPException(status_code=500, detail="Collector not initialized")
        
        users_per_batch = request.users_per_batch
        samples_per_user = request.samples_per_user
        
        
        # Sample user data
        user_data = data_sampler(users_per_batch=users_per_batch, samples_per_user=samples_per_user, device=device)
        
        # Compute reward matrix for all attributes
        R = await compute_rewards(user_data, len(attribute_prompts))  # [batch_size, d]
        
        collections_count += 1
        
        # Convert tensors to lists for JSON serialization
        response = CollectionResponse(
            R=R.detach().cpu().tolist(),
            user_data=user_data,
            success=True
        )
        
        
        return response
        
    except Exception as e:
        logging.error(f"Error in collect_batch: {e}")
        return CollectionResponse(
            R=[], user_data={},
            success=False, error=str(e)
        )

@app.get("/status")
async def get_status():
    return {
        "status": "running" if data_sampler else "initializing",
        "collections_served": collections_count
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="Collector Server")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model-name", type=str, required=True, help="VLLM model name")
    parser.add_argument("--vllm-server-url", type=str, required=True, help="VLLM server URL")
    
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
    initialize_collector(
        d=args.d,
        dataset_path=args.dataset_path,
        device_str=args.device,
        attribute_prompts_path=args.attribute_prompts_path,
        vllm_server_url=args.vllm_server_url,
        model_name_arg=args.model_name
    )
    
    logging.info(f"Starting Collector Server on {args.host}:{args.port}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()