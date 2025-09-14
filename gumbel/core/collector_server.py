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
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

# ---- Local imports ----
from .sampler import DataSampler
from ..utils import async_utils

# =========================
# Request/Response models
# =========================
class CollectionRequest(BaseModel):
    users_per_batch: int
    samples_per_user: int
    behavior_logits: List[float] = []  # Behavioral policy logits (optional)
    tau: float = 1.0  # Temperature for sampling (optional)

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
# app will be defined later with lifespan

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
REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "512"))  # Process requests in batches of this size

async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """
    Drift reward = attr_avg_logprob - base_avg_logprob  (shape [B, d])
    """
    prompts: List[str] = user_data["prompts"]
    outputs: List[str] = user_data["outputs"]
    
    return await async_utils.compute_drift_rewards(
        session=http_session,
        tokenizer=tokenizer,
        prompts=prompts,
        outputs=outputs,
        base_prompt=base_prompt,
        attribute_prompts=attribute_prompts,
        vllm_url=f"{vllm_server_url}/v1/completions",
        model_id=model_name,
        device=device
    )

# FastAPI endpoints will be defined in main() after app is created

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global http_session, sem
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    sem = asyncio.Semaphore(CONCURRENCY)
    
    yield
    
    # Shutdown
    if http_session is not None:
        await http_session.close()
        http_session = None

# Update app initialization
app = FastAPI(lifespan=lifespan)

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

def main():
    global app
    
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

    # Initialize collector (everything except aiohttp session)
    initialize_collector(
        d=args.d,
        dataset_path=args.dataset_path,
        device_str=args.device,
        attribute_prompts_path=args.attribute_prompts_path,
        vllm_server_url_arg=args.vllm_server_url,
        model_name_arg=args.model_name,
    )

    # Create app with lifespan
    app = FastAPI(lifespan=lifespan)
    
    # Re-register routes (since app was recreated)
    @app.post("/generate_batch", response_model=CollectionResponse)
    async def generate_batch(request: CollectionRequest):
        global collections_count
        try:
            if data_sampler is None:
                raise HTTPException(status_code=500, detail="Collector not initialized")

            # Sample user data (currently ignores behavior_logits and tau)
            # Future: could use behavior_logits to bias user/prompt selection
            user_data = data_sampler(
                users_per_batch=request.users_per_batch,
                samples_per_user=request.samples_per_user,
                device=device
            )

            # Compute drift rewards using current behavioral policy parameters
            R = await compute_rewards(user_data, len(attribute_prompts))
            collections_count += 1

            # Log behavioral policy info if provided
            if request.behavior_logits:
                logging.debug(f"Received behavioral policy: {len(request.behavior_logits)} logits, tau={request.tau:.3f}")

            return CollectionResponse(
                R=R.detach().cpu().tolist(),
                user_data=user_data,
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

    logging.info(f"Starting Collector Server on {args.host}:{args.port} | CONCURRENCY={CONCURRENCY}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())

if __name__ == "__main__":
    main()
