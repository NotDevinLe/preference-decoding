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
from src.core.drift import RewardModel
import random
import numpy as np

attribute_prompts: Optional[List[str]] = None
base_prompt: str = "You are a helpful assistant."

device: Optional[torch.device] = None
vllm_server_url: Optional[str] = None
model_name: Optional[str] = None
tokenizer: Optional[AutoTokenizer] = None

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
        attribute_prompts_local = loaded_prompts
    elif isinstance(loaded_prompts, dict) and "prompts" in loaded_prompts:
        attribute_prompts_local = loaded_prompts["prompts"]
    else:
        raise ValueError("Invalid attribute prompts file format")

    attribute_prompts = attribute_prompts_local

async def main():
    parser = argparse.ArgumentParser(description="Reward Matrix Computation Script")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file", default="configs/experiment.yaml")
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

    reward_model = RewardModel(
        model_name=model_name,
        tokenizer=tokenizer,
        base_prompt=base_prompt,
        attribute_prompts=attribute_prompts,
        vllm_server_url=vllm_url,
        device=device_str,
        max_concurrent_requests=50,
        max_retries=10,
        request_timeout=60,
        request_batch_size=1,
    )

    try:
        for i in range(25):
            dataset_path = f'data/processed_prism/user{i}.json'

            if not os.path.exists(dataset_path):
                logging.info(f"User {i} data not found, skipping")
                continue

            logging.info(f"Processing user {i} from {dataset_path}")
            
            with open(dataset_path, 'r') as f:
                user_data = json.load(f)
            
            await reward_model.compute_rewards(user_data, i, split="train", save_dir=f'eval_rewards/llama1b/prism', batch_size=1)
    except Exception as e:
        logging.exception("Error in reward computation")
        raise

if __name__ == "__main__":
    asyncio.run(main())