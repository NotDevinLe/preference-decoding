#!/usr/bin/env python3
"""
Collector Server: samples data and computes drift rewards using a remote vLLM server.
Optimized for throughput:
  - Reuses one aiohttp.ClientSession (keep-alive)
  - Sends (d+1)*B requests in one large concurrent wave
  - Uses /v1/completions with echo=True, max_tokens=0, logprobs=1 (no generation)
"""

import os
import json
import torch
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Tuple

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

# ---- Local imports ----
from sampler import DataSampler
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import async_utils

# =========================
# Request/Response models
# =========================
class CollectionRequest(BaseModel):
    users_per_batch: int
    samples_per_user: int

class CollectionResponse(BaseModel):
    R: List[List[float]]              # [batch_size, d]
    user_data: Dict[str, Any]
    success: bool
    error: str | None = None

class StatusResponse(BaseModel):
    status: str
    collections_served: int

# =========================
# Globals
# =========================
app = FastAPI()

data_sampler: DataSampler | None = None
attribute_prompts: List[str] | None = None
base_prompt: str = "You are a helpful assistant."
collections_count: int = 0

device: torch.device | None = None
vllm_server_url: str | None = None
model_name: str | None = None
tokenizer: AutoTokenizer | None = None

http_session: aiohttp.ClientSession | None = None
sem: asyncio.Semaphore | None = None
CONCURRENCY = int(os.getenv("COLLECTOR_CONCURRENCY", "256"))  # tune to GPU/server (128–512 common)

async def get_log_probs_batch(system_prompts: List[str], user_prompts: List[str], completion_texts: List[str]) -> Tuple[List[float], List[int]]:
    """Get log probabilities using async_utils with collector's VLLM server"""
    # Temporarily override async_utils globals
    original_vllm_url = async_utils.VLLM_URL
    original_model_id = async_utils.MODEL_ID
    
    try:
        async_utils.VLLM_URL = f"{vllm_server_url}/v1/completions"
        async_utils.MODEL_ID = model_name
        
        return await async_utils.get_log_probs_async(http_session, tokenizer, system_prompts, user_prompts, completion_texts)
    finally:
        async_utils.VLLM_URL = original_vllm_url
        async_utils.MODEL_ID = original_model_id

async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """
    Drift reward = attr_avg_logprob - base_avg_logprob  (shape [B, d])
    """
    prompts: List[str] = user_data["prompts"]
    outputs: List[str] = user_data["outputs"]
    B = len(outputs)
    if B == 0:
        return torch.zeros(0, d, device=device)

    # Get base log probabilities
    base_probs, base_counts = await get_log_probs_batch([base_prompt] * B, prompts, outputs)
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Build reward matrix
    reward_matrix = torch.zeros(B, d, device=device)
    
    # Compute drift scores for each attribute
    for attr_idx in range(d):
        attr_prompt = attribute_prompts[attr_idx]
        attr_probs, attr_counts = await get_log_probs_batch([attr_prompt] * B, prompts, outputs)
        attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        reward_matrix[:, attr_idx] = attr_scores - base_scores
    
    return reward_matrix

# =========================
# FastAPI endpoints
# =========================
@app.post("/generate_batch", response_model=CollectionResponse)
async def generate_batch(request: CollectionRequest):
    global collections_count
    try:
        if data_sampler is None:
            raise HTTPException(status_code=500, detail="Collector not initialized")

        # Sample user data
        user_data = data_sampler(
            users_per_batch=request.users_per_batch,
            samples_per_user=request.samples_per_user,
            device=device
        )

        # Compute reward matrix for all attributes (single big wave)
        R = await compute_rewards(user_data, len(attribute_prompts))  # [B, d]
        collections_count += 1

        # Return tensors as lists (JSON-serializable)
        return CollectionResponse(
            R=R.detach().cpu().tolist(),
            user_data=user_data,  # assumed JSON-serializable by your sampler
            success=True
        )
    except Exception as e:
        logging.exception("Error in /generate_batch")
        return CollectionResponse(R=[], user_data={}, success=False, error=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        status="running" if data_sampler else "initializing",
        collections_served=collections_count
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("shutdown")
async def _shutdown():
    global http_session
    if http_session is not None:
        await http_session.close()
        http_session = None

# =========================
# Initialization & main
# =========================
def initialize_collector(
    d: int,
    dataset_path: str,
    device_str: str,
    attribute_prompts_path: str,
    vllm_server_url_arg: str,
    model_name_arg: str,
):
    global data_sampler, attribute_prompts, device, vllm_server_url, model_name, tokenizer
    global http_session, sem

    device = torch.device(device_str)
    data_sampler = DataSampler(dataset_path=dataset_path)

    vllm_server_url = vllm_server_url_arg.rstrip("/")
    model_name = model_name_arg

    tokenizer = AutoTokenizer.from_pretrained(model_name_arg)
    tokenizer.pad_token = tokenizer.eos_token

    with open(attribute_prompts_path, "r") as f:
        loaded_prompts = json.load(f)

    if isinstance(loaded_prompts, list):
        attribute_prompts_local = loaded_prompts[:d]
    elif isinstance(loaded_prompts, dict) and "prompts" in loaded_prompts:
        attribute_prompts_local = loaded_prompts["prompts"][:d]
    else:
        raise ValueError("Invalid attribute prompts file format")

    if len(attribute_prompts_local) < d:
        raise ValueError(f"Need at least {d} attribute prompts")
    attribute_prompts = attribute_prompts_local

    # Create a single shared session + semaphore
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)  # no per-host cap; sem gates concurrency
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    sem = asyncio.Semaphore(CONCURRENCY)

def main():
    parser = argparse.ArgumentParser(description="Collector Server (optimized)")
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument("--model-name", type=str, required=True, help="HF model id (for tokenizer)")
    parser.add_argument("--vllm-server-url", type=str, required=True, help="Base URL of vLLM server (e.g. http://localhost:8000)")
    parser.add_argument("--attribute-prompts-path", type=str, required=True, help="Path to attribute prompts JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for torch tensors")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    initialize_collector(
        d=args.d,
        dataset_path=args.dataset_path,
        device_str=args.device,
        attribute_prompts_path=args.attribute_prompts_path,
        vllm_server_url_arg=args.vllm_server_url,
        model_name_arg=args.model_name,
    )

    logging.info(f"Starting Collector Server on {args.host}:{args.port} | CONCURRENCY={CONCURRENCY}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())

if __name__ == "__main__":
    main()
