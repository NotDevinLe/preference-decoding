#!/usr/bin/env python3

import os
import json
import math
import time
import asyncio
import logging
import argparse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple, Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoTokenizer

from src.core.sampler import DataSampler

class CollectionRequest(BaseModel):
    users_per_batch: int
    samples_per_user: int

class CollectionResponse(BaseModel):
    R: List[List[float]]
    user_data: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    collections_served: int

data_sampler: Optional[DataSampler] = None
attribute_prompts: Optional[List[str]] = None
base_prompt: str = "You are a helpful assistant."
collections_count: int = 0

device: Optional[torch.device] = None
vllm_server_url: Optional[str] = None
model_name: Optional[str] = None
tokenizer: Optional[AutoTokenizer] = None

http_session: Optional[ClientSession] = None
sem: Optional[asyncio.Semaphore] = None

CONCURRENCY = int(os.getenv("COLLECTOR_CONCURRENCY", "256"))         # max in-flight POSTs
REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "512"))      # size of coroutines launched at once
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))              # retry count for resets/disconnects
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))          # seconds
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", str(CONCURRENCY)))   # per-host socket cap
TOTAL_CONN_LIMIT = int(os.getenv("TOTAL_CONN_LIMIT", str(CONCURRENCY * 2)))

def build_full_prompt_and_ids(tokenizer, sys_prompt: str, user_prompt: str, completion: str):
    """
    Returns:
      full_text, n_prefix_tokens, completion_len_tokens, prompt_ids, comp_ids
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt.strip()},
         {"role": "user",   "content": user_prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    return prompt_text + completion, len(prompt_ids), len(comp_ids), prompt_ids, comp_ids

def sum_completion_logprobs(resp_json: Dict[str, Any], n_prefix: int, comp_len: int) -> float:
    lp = resp_json["choices"][0]["logprobs"]["token_logprobs"]
    end = min(len(lp), n_prefix + comp_len)   # guard if server ever adds a token
    seg = [x for x in lp[n_prefix:end] if x is not None]
    return float(sum(seg))

async def post_json_with_retry(
    session: ClientSession,
    url: str,
    payload: Dict[str, Any],
    retries: int = REQUEST_RETRIES,
    base_delay: float = 0.2,
) -> Dict[str, Any]:
    """
    Keep spamming Posts for some time until it succeeds or runs out of tries
    """
    last_exception = None
    
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload) as r:
                if r.status == 0:
                    raise aiohttp.ClientOSError(f"HTTP status 0 - connection failed")
                r.raise_for_status()
                return await r.json()
        except (aiohttp.ClientConnectionResetError,
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientConnectorError,
                asyncio.TimeoutError) as e:
            last_exception = e
            if attempt == retries - 1:
                logging.error(f"HTTP REQUEST FAILED after {retries} attempts: {type(e).__name__}: {e}")
                raise
            retry_delay = base_delay * (2 ** attempt)
            logging.warning(f"HTTP ERROR (attempt {attempt + 1}/{retries}): {type(e).__name__}: {e} - retrying in {retry_delay:.1f}s")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logging.error(f"NON-RETRYABLE HTTP ERROR: {type(e).__name__}: {e}")
            raise
    
    # Should never reach here, but just in case
    if last_exception is not None:
        raise last_exception
    else:
        raise Exception("Unknown HTTP failure")

async def score_many(
    session: ClientSession,
    url: str,
    payloads: List[Dict[str, Any]],
    batch_size: int,
) -> List[Any]:
    """
    Launch many POSTs in chunks (to keep memory sane) and with a global semaphore.
    Returns a list of responses (or exceptions) matching payload order.
    """
    results: List[Any] = [None] * len(payloads)
    global sem

    async def one(i: int, body: Dict[str, Any]):
        async with sem:
            return await post_json_with_retry(session, url, body)

    # chunking the task creation mitigates large coroutine fan-out
    total_batches = math.ceil(len(payloads) / batch_size)
    for batch_idx, start in enumerate(range(0, len(payloads), batch_size), 1):
        end = min(len(payloads), start + batch_size)
        chunk = payloads[start:end]
        logging.info(f"PROCESSING BATCH {batch_idx}/{total_batches}: requests {start+1}-{end} ({len(chunk)} requests)")
        tasks = [asyncio.create_task(one(start + j, body)) for j, body in enumerate(chunk)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        results[start:end] = chunk_results
        logging.info(f"COMPLETED BATCH {batch_idx}/{total_batches}")

    return results

async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """
    Drift reward = attr_avg_logprob - base_avg_logprob  (shape [B, d])
    Builds ALL (d+1)*B requests and dispatches them with bounded concurrency + retries.
    """
    global http_session
    
    # Validate components
    if tokenizer is None or vllm_server_url is None or model_name is None:
        raise RuntimeError("Collector not properly initialized")
    
    # Check/recreate session if needed
    if http_session is None or http_session.closed:
        logging.warning("HTTP SESSION: Recreating closed session...")
        connector = aiohttp.TCPConnector(
            limit=TOTAL_CONN_LIMIT,
            limit_per_host=PER_HOST_LIMIT,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )
        timeout = ClientTimeout(total=REQUEST_TIMEOUT)
        http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        logging.info("HTTP SESSION: Recreated")

    prompts: List[str] = user_data["prompts"]
    outputs: List[str] = user_data["outputs"]
    B = len(outputs)
    d_attrs = d

    # Pre-build payloads and per-request metadata for reconstruction
    payloads: List[Dict[str, Any]] = []
    metas: List[Tuple[int, int, int]] = []  # (sample_idx, attr_idx(-1 for base), comp_len_subset)

    # Helper to build an echo-scoring payload
    def mk_payload(full_prompt: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,  # keep 1; if your vLLM build supports chosen-logprob without top-k, you can set 0
        }

    # We also keep n_prefix/comp_len for slicing logprobs; recompute cheaply after
    prefix_and_len: List[Tuple[int, int]] = []

    # Base first, then all attributes per sample
    for i in range(B):
        # Base
        full_prompt, n_prefix, comp_len, prompt_ids, comp_ids = build_full_prompt_and_ids(
            tokenizer, base_prompt, prompts[i], outputs[i]
        )
        payloads.append(mk_payload(full_prompt))
        metas.append((i, -1, comp_len))
        prefix_and_len.append((n_prefix, comp_len))

        # Attributes
        for a_idx, a_sys in enumerate(attribute_prompts):
            full_prompt, n_prefix, comp_len, prompt_ids, comp_ids = build_full_prompt_and_ids(
                tokenizer, a_sys, prompts[i], outputs[i]
            )
            payloads.append(mk_payload(full_prompt))
            metas.append((i, a_idx, comp_len))
            prefix_and_len.append((n_prefix, comp_len))

    # Fire requests with timeout protection
    url = f"{vllm_server_url}/v1/completions"
    total_requests = len(payloads)
    logging.info(f"VLLM REQUESTS: Starting {total_requests} requests to {url}")
    
    start_time = time.time()
    try:
        raw_results = await asyncio.wait_for(
            score_many(
                session=http_session,
                url=url,
                payloads=payloads,
                batch_size=REQUEST_BATCH_SIZE,
            ),
            timeout=REQUEST_TIMEOUT * 2  # Give extra time for large batches
        )
        elapsed = time.time() - start_time
        logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logging.error(f"VLLM REQUESTS: Timed out after {elapsed:.1f}s")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"VLLM REQUESTS: Failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        raise

    # Prepare tensors
    base_scores = torch.zeros(B, dtype=torch.float32, device=device)
    base_counts = torch.zeros(B, dtype=torch.float32, device=device)
    attr_scores = torch.zeros(B, d_attrs, dtype=torch.float32, device=device)
    attr_counts = torch.zeros(B, d_attrs, dtype=torch.float32, device=device)

    # Parse results; on failures, leave zeros (or you can raise)
    for idx, res in enumerate(raw_results):
        i, a_idx, comp_len = metas[idx]
        n_prefix, comp_len_chk = prefix_and_len[idx]
        # guard mismatches
        comp_len_eff = min(comp_len, comp_len_chk)

        if isinstance(res, Exception):
            logging.warning(f"Score failed for sample={i}, attr={a_idx}: {type(res).__name__}: {res}")
            continue

        try:
            s = sum_completion_logprobs(res, n_prefix, comp_len_eff)
            if a_idx == -1:
                base_scores[i] += s
                base_counts[i] += max(1, comp_len_eff)
            else:
                attr_scores[i, a_idx] += s
                attr_counts[i, a_idx] += max(1, comp_len_eff)
        except Exception as e:
            logging.warning(f"Parse failed for sample={i}, attr={a_idx}: {e}")

    # Average and compute drift reward
    # Avoid div-by-zero; if a row failed entirely, it stays 0
    base_avg = torch.where(base_counts > 0, base_scores / base_counts, torch.zeros_like(base_scores))
    attr_avg = torch.where(attr_counts > 0, attr_scores / torch.clamp_min(attr_counts, 1.0), torch.zeros_like(attr_scores))

    reward_matrix = attr_avg - base_avg[:, None]
    return reward_matrix

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_session, sem
    # Bounded connector + cleanup of closed sockets to avoid stale keep-alives
    connector = aiohttp.TCPConnector(
        limit=TOTAL_CONN_LIMIT,
        limit_per_host=PER_HOST_LIMIT,
        enable_cleanup_closed=True,
        ttl_dns_cache=300,
    )
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    sem = asyncio.Semaphore(CONCURRENCY)
    try:
        yield
    finally:
        if http_session is not None and not http_session.closed:
            await http_session.close()

app = FastAPI(lifespan=lifespan)

def initialize_collector(
    d: int,
    dataset_path: str,
    device_str: str,
    attribute_prompts_path: str,
    vllm_server_url_arg: str,
    model_name_arg: str,
):
    global data_sampler, attribute_prompts, device, vllm_server_url, model_name, tokenizer

    device = torch.device(device_str)
    data_sampler = DataSampler(dataset_path=dataset_path)

    vllm_server_url = vllm_server_url_arg.rstrip("/")
    model_name = model_name_arg

    tokenizer = AutoTokenizer.from_pretrained(model_name_arg)
    # Ensure pad token exists for some tokenizers
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
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

    # Store
    attribute_prompts = attribute_prompts_local

def main():
    global app
    
    parser = argparse.ArgumentParser(description="Collector Server (optimized)")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--d", type=int, help="Number of attributes")
    parser.add_argument("--dataset-path", type=str, help="Dataset path")
    parser.add_argument("--model-name", type=str, help="HF model id (for tokenizer)")
    parser.add_argument("--vllm-server-url", type=str, help="Base URL of vLLM server")
    parser.add_argument("--attribute-prompts-path", type=str, help="Path to attribute prompts JSON")
    parser.add_argument("--host", type=str, help="Bind host")
    parser.add_argument("--port", type=int, help="Bind port")
    parser.add_argument("--device", type=str, help="Device for torch tensors")
    parser.add_argument("--log-level", type=str, help="Logging level")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        try:
            from ..utils.config_loader import load_config, ConfigLoader
            config = load_config(args.config)
            collector_config = ConfigLoader.get_collector_config(config)
            
            d = int(args.d or collector_config["d"])
            dataset_path = str(args.dataset_path or collector_config["dataset_path"])
            model_name = str(args.model_name or collector_config["model_name"])
            vllm_url = str(args.vllm_server_url or collector_config["vllm_server_url"])
            attribute_prompts_path = str(args.attribute_prompts_path or collector_config["attribute_prompts_path"])
            host = str(args.host or collector_config["host"])
            port = int(args.port or collector_config["port"])
            device_str = str(args.device or collector_config["device"])
            log_level = str(args.log_level or collector_config["log_level"])
            
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logging.info(f"Loaded collector config from {args.config}")
        except Exception as e:
            logging.error(f"Failed to load config from {args.config}: {e}")
            return
    else:
        # Require individual args when no config
        if not all([args.dataset_path, args.model_name, args.vllm_server_url, args.attribute_prompts_path]):
            parser.error("Either --config or all of --dataset-path, --model-name, --vllm-server-url, --attribute-prompts-path must be provided")
        
        d = int(args.d or 100)
        dataset_path = str(args.dataset_path)
        model_name = str(args.model_name)
        vllm_url = str(args.vllm_server_url)
        attribute_prompts_path = str(args.attribute_prompts_path)
        host = str(args.host or "0.0.0.0")
        port = int(args.port or 8001)
        device_str = str(args.device or "cuda:0")
        log_level = str(args.log_level or "INFO")

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Initialize collector
    initialize_collector(
        d=d,
        dataset_path=dataset_path,
        device_str=device_str,
        attribute_prompts_path=attribute_prompts_path,
        vllm_server_url_arg=vllm_url,
        model_name_arg=model_name,
    )

    # Recreate app with lifespan
    app = FastAPI(lifespan=lifespan)
    
    @app.post("/generate_batch", response_model=CollectionResponse)
    async def generate_batch(request: CollectionRequest):
        global collections_count
        try:
            if data_sampler is None:
                raise HTTPException(status_code=500, detail="Collector not initialized")

            # Sample
            user_data = data_sampler(
                users_per_batch=request.users_per_batch,
                samples_per_user=request.samples_per_user,
                device=device
            )

            # Compute
            R = await compute_rewards(user_data, len(attribute_prompts))
            collections_count += 1

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

    logging.info(
        f"Starting Collector Server on {host}:{port} | "
        f"CONCURRENCY={CONCURRENCY} | BATCH_SIZE={REQUEST_BATCH_SIZE} | "
        f"PER_HOST_LIMIT={PER_HOST_LIMIT} | TOTAL_CONN_LIMIT={TOTAL_CONN_LIMIT}"
    )

    # Single worker
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(), workers=1)

if __name__ == "__main__":
    main()
