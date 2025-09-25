#!/usr/bin/env python3

import os
import json
import math
import time
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
import aiohttp
import torch
from transformers import AutoTokenizer
from src.core.drift import compute_rewards, build_full_prompt, sum_completion_logprobs

attribute_prompts: Optional[List[str]] = None
base_prompt: str = "You are a helpful assistant."

device: Optional[torch.device] = None
vllm_server_url: Optional[str] = None
model_name: Optional[str] = None
tokenizer: Optional[AutoTokenizer] = None

REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "1024"))
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3")) 
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))

async def _send_request(session: aiohttp.ClientSession, server_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single request to the server (VLLM or gateway)."""
    url = f"{server_url}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
    except Exception as e:
        return e 

async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """
    Drift reward = attr_avg_logprob - base_avg_logprob  (shape [B, d])
    Uses LiteRegistry for efficient server management and load balancing
    """
    # Validate components
    if tokenizer is None or vllm_server_url is None or model_name is None:
        raise RuntimeError("Collector not properly initialized")

    prompts: List[str] = [example["prompt"] for example in user_data]
    outputs: List[str] = [example["chosen"] for example in user_data]
    B = len(outputs)
    d_attrs = d

    payloads: List[Dict[str, Any]] = []
    metas: List[Tuple[int, int, int]] = []
    prefix_and_len: List[Tuple[int, int]] = []

    for i in range(B):
        # Base prompt
        full_prompt, prefix_len, comp_len = build_full_prompt(
            tokenizer, base_prompt, prompts[i], outputs[i]
        )
        payloads.append({
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        })
        metas.append((i, -1, comp_len))
        prefix_and_len.append((prefix_len, comp_len))

        for a_idx, a_sys in enumerate(attribute_prompts):
            full_prompt, prefix_len, comp_len = build_full_prompt(
                tokenizer, a_sys, prompts[i], outputs[i]
            )
            payloads.append({
                "model": model_name,
                "prompt": full_prompt,
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            })
            metas.append((i, a_idx, comp_len))
            prefix_and_len.append((prefix_len, comp_len))
    
    total_requests = len(payloads)
    logging.info(f"VLLM REQUESTS: Starting {total_requests} requests via Gateway")
    start_time = time.time()
    
    # Send requests directly to gateway
    async with aiohttp.ClientSession() as session:
        logging.info(f"Sending {len(payloads)} requests to gateway...")
        
        # Process requests in batches to avoid overwhelming the server
        batch_size = REQUEST_BATCH_SIZE
        raw_results = []
        
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i+batch_size]
            batch_tasks = []
            
            for payload in batch:
                batch_tasks.append(_send_request(session, vllm_server_url, payload))
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            raw_results.extend(batch_results)
            
            logging.info(f"Completed batch {i//batch_size + 1}/{(len(payloads) + batch_size - 1)//batch_size}")
        
        logging.info("Gateway batch processing completed")
    
    elapsed = time.time() - start_time
    logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")

    base_scores = torch.zeros(B, dtype=torch.float32, device="cpu")
    base_counts = torch.zeros(B, dtype=torch.float32, device="cpu")
    attr_scores = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")
    attr_counts = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")

    for idx, res in enumerate(raw_results):
        i, a_idx, comp_len = metas[idx]
        prefix_len, comp_len_chk = prefix_and_len[idx]
        comp_len_eff = min(comp_len, comp_len_chk)

        if isinstance(res, Exception):
            logging.warning(f"Score failed for sample={i}, attr={a_idx}: {type(res).__name__}: {res}")
            continue

        try:
            s = sum_completion_logprobs(res, prefix_len, comp_len_eff)
            if a_idx == -1:
                base_scores[i] += s
                base_counts[i] += max(1, comp_len_eff)
            else:
                attr_scores[i, a_idx] += s
                attr_counts[i, a_idx] += max(1, comp_len_eff)
        except Exception as e:
            logging.warning(f"Parse failed for sample={i}, attr={a_idx}: {e}")

    base_avg = torch.where(base_counts > 0, base_scores / base_counts, torch.zeros_like(base_scores))
    attr_avg = torch.where(attr_counts > 0, attr_scores / torch.clamp_min(attr_counts, 1.0), torch.zeros_like(attr_scores))
    reward_matrix = attr_avg - base_avg[:, None]
    
    return reward_matrix

def initialize_collector(
    d: int,
    device_str: str,
    attribute_prompts_path: str,
    vllm_server_url_arg: str,
    model_name_arg: str,
):
    global attribute_prompts, device, vllm_server_url, model_name, tokenizer

    device = torch.device(device_str)

    vllm_server_url = vllm_server_url_arg.rstrip("/")
    model_name = model_name_arg

    tokenizer = AutoTokenizer.from_pretrained(model_name_arg)
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

    attribute_prompts = attribute_prompts_local

async def main():
    parser = argparse.ArgumentParser(description="Reward Matrix Computation Script")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file", default="gumbel/configs/experiment.yaml")
    parser.add_argument("--output-path", type=str, default="rewards.pt", help="Output path for reward matrix")
    args = parser.parse_args()
    
    try:
        from utils.config_loader import load_config, ConfigLoader
        config = load_config(args.config)
        collector_config = ConfigLoader.get_collector_config(config)
        
        d = int(collector_config["d"])
        model_name = str(collector_config["model_name"])
        vllm_url = str(collector_config["vllm_server_url"])
        attribute_prompts_path = str(collector_config["attribute_prompts_path"])
        device_str = str(collector_config["device"])
        log_level = str(collector_config["log_level"])
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Loaded collector config from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load config from {args.config}: {e}")
        return

    initialize_collector(
        d=d,
        device_str=device_str,
        attribute_prompts_path=attribute_prompts_path,
        vllm_server_url_arg=vllm_url,
        model_name_arg=model_name,
    )

    # Test server connection
    try:
        async with aiohttp.ClientSession() as test_session:
            async with test_session.get(f"{vllm_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logging.info(f"Server: {vllm_url} - HEALTHY")
                else:
                    logging.warning(f"Server: {vllm_url} - UNHEALTHY (status {resp.status})")
    except Exception as e:
        logging.warning(f"Server: {vllm_url} - UNREACHABLE ({type(e).__name__}: {e})")


    total_rewards = []
    try:
        for i in range(11, 200):
            dataset_path = f'data/persona_pref/user{i}_train.json'
            logging.info(f"Processing user {i} from {dataset_path}")
            
            with open(dataset_path, 'r') as f:
                user_data = json.load(f)
            
            R_batch = await compute_rewards(user_data, len(attribute_prompts))
            total_rewards.append(R_batch)
            
    except Exception as e:
        logging.exception("Error in reward computation")
        raise

    total_rewards = torch.cat(total_rewards, dim=0)
    torch.save(total_rewards, "rewards_all.pt")
    logging.info(f"Saved total reward matrix with shape {total_rewards.shape} to rewards_all.pt")

if __name__ == "__main__":
    asyncio.run(main())