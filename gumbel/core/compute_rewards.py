#!/usr/bin/env python3

import os
import json
import math
import time
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
from literegistry import RegistryHTTPClient, FileSystemKVStore, RegistryClient
import torch
from transformers import AutoTokenizer

attribute_prompts: Optional[List[str]] = None
base_prompt: str = "You are a helpful assistant."

device: Optional[torch.device] = None
vllm_server_url: Optional[str] = None
model_name: Optional[str] = None
tokenizer: Optional[AutoTokenizer] = None

httpClient: Optional[RegistryHTTPClient] = None
fileSystemKVStore: Optional[FileSystemKVStore] = None
registryClient: Optional[RegistryClient] = None

REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "512"))      # size of coroutines launched at once
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))              # retry count for resets/disconnects
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))          # seconds

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

# Removed post_json_with_retry and score_many functions - using LiteRegistry instead

async def compute_rewards(user_data: Dict[str, Any], d: int) -> torch.Tensor:
    """
    Drift reward = attr_avg_logprob - base_avg_logprob  (shape [B, d])
    Builds ALL (d+1)*B requests and dispatches them with bounded concurrency + retries.
    """
    # Validate components
    if tokenizer is None or vllm_server_url is None or model_name is None:
        raise RuntimeError("Collector not properly initialized")

    prompts: List[str] = [example["prompt"] for example in user_data]
    outputs: List[str] = [example["chosen"] for example in user_data]
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
    
    total_requests = len(payloads)
    logging.info(f"VLLM REQUESTS: Starting {total_requests} requests via LiteRegistry")
    
    start_time = time.time()
    try:
        # Check if any servers are available first
        available_servers = await registryClient.get_all(model_name)
        if not available_servers:
            raise RuntimeError(f"No servers available for model {model_name}. Please start some vLLM servers first.")
        
        logging.info(f"Found {len(available_servers)} servers for model {model_name}: {available_servers}")
        
        # Create registry client for the model
        async with RegistryHTTPClient(
            registry=registryClient,
            value=model_name,
            max_parallel_requests=REQUEST_BATCH_SIZE,
            timeout=REQUEST_TIMEOUT,
            max_retries=REQUEST_RETRIES
        ) as httpClient:
            # Dispatch all requests in batches with proper server rotation
            raw_results: List[Any] = []
            for i in range(0, len(payloads), REQUEST_BATCH_SIZE):
                batch_payloads = payloads[i:i + REQUEST_BATCH_SIZE]
                
                # Use different starting server indices for each request in the batch
                batch_results = await asyncio.gather(*[
                    httpClient.request_with_rotation("v1/completions", payload, initial_server_idx=j % len(available_servers))
                    for j, payload in enumerate(batch_payloads)
                ])
                raw_results.extend([result for result, _ in batch_results])
        
        elapsed = time.time() - start_time
        logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")
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

def save_rewards(rewards: torch.Tensor, path: str):
    torch.save(rewards, path)

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

    # Initialize registry components
    global fileSystemKVStore, registryClient
    fileSystemKVStore = FileSystemKVStore("/gscratch/ark/devinl6/registry")
    registryClient = RegistryClient(fileSystemKVStore, service_type="model_path")

async def main():
    parser = argparse.ArgumentParser(description="Reward Matrix Computation Script")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file", default="gumbel/configs/experiment.yaml")
    parser.add_argument("--output-path", type=str, default="rewards.pt", help="Output path for reward matrix")
    args = parser.parse_args()
    
    try:
        from ..utils.config_loader import load_config, ConfigLoader
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

    # Initialize collector
    initialize_collector(
        d=d,
        device_str=device_str,
        attribute_prompts_path=attribute_prompts_path,
        vllm_server_url_arg=vllm_url,
        model_name_arg=model_name,
    )

# Test connectivity to each server
    available_servers = await registryClient.get_all(model_name)
    for i, server in enumerate(available_servers):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as test_session:
                async with test_session.get(f"{server}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logging.info(f"Server {i+1}: {server} - HEALTHY")
                    else:
                        logging.warning(f"Server {i+1}: {server} - UNHEALTHY (status {resp.status})")
        except Exception as e:
            logging.warning(f"Server {i+1}: {server} - UNREACHABLE ({type(e).__name__}: {e})")

    try:
        for i in range(11, 13):
            dataset_path = f'data/persona_pref/user{i}_train.json'
            logging.info(f"Processing user {i} from {dataset_path}")
            
            with open(dataset_path, 'r') as f:
                user_data = json.load(f)
            
            # Compute rewards for this user
            R_batch = await compute_rewards(user_data, len(attribute_prompts))
            
            # Save results for this user
            user_output_path = f"rewards_user{i}.pt"
            save_rewards(R_batch, user_output_path)
            logging.info(f"Saved reward matrix for user {i} with shape {R_batch.shape} to {user_output_path}")
            
    except Exception as e:
        logging.exception("Error in reward computation")
        raise

if __name__ == "__main__":
    asyncio.run(main())
