import asyncio
import os
import time
import logging
import psutil
import resource
import gc
import signal
import traceback
import sys
from typing import Tuple, List, Dict, Any, Union, Optional
import torch
import numpy as np
from transformers import AutoTokenizer
import aiohttp
import pathlib

class RewardModel:

    """

    Class for computing rewards from weighted average of log probabilities.

    Handles Async requests efficiently with VLLM compatible gateway.

    Key Functions:
        - compute_rewards: Computes rewards for a batch of user data
        - approximate: Learns a attribute weight vector from a dataset
        - evaluate_accuracy: Evaluates the accuracy of the attribute weight vector on a test dataset
        -log_probs: Computes log probabilities for a batch of user data

    """

    def __init__(self, model_name:str, tokenizer:AutoTokenizer, base_prompt:str, attribute_prompts:List[str], vllm_server_url:str, request_timeout:float=10, request_batch_size:int=500, max_retries:int=3, max_concurrent_requests:int=100, device:str="cpu"):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.vllm_server_url = vllm_server_url
        self.device = device

        self.request_timeout = request_timeout
        self.request_batch_size = request_batch_size
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
    
    # Some cool helper functions
    def _build_full_prompt(self, tokenizer, sys_prompt: str, user_prompt: str, completion: str) -> Tuple[str, int, int]:
        """Return: full_text (prompt+completion), prefix_tokens, completion_tokens"""
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_prompt.strip()},
            {"role": "user",   "content": user_prompt.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        return prompt_text + completion, len(prompt_ids), len(comp_ids)
    
    def _build_full_prompt_multi_turn(self, tokenizer, sys_prompt: str, user_prompts: List[str], completion: str) -> Tuple[str, int, int]:
        """Pass in the prompt value of the prism dataset for user_prompts don't change anything okay?"""

        conversation = [{"role": "system", "content": sys_prompt.strip()}]
        for prompt in user_prompts:
            conversation.append({"role": prompt["role"], "content": prompt["content"].strip()})

        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        return prompt_text + completion, len(prompt_ids), len(comp_ids)

    @staticmethod
    def sum_completion_logprobs(resp_json, prefix_len: int, comp_len: int) -> float:
        lp = resp_json["choices"][0]["logprobs"]["token_logprobs"]
        end = min(len(lp), prefix_len + comp_len)
        seg = [x for x in lp[prefix_len:end] if x is not None]
        return float(sum(seg))


    async def _make_vllm_request(self, session: aiohttp.ClientSession, gateway_url: str, payload: Dict) -> Dict:
        async with session.post(f"{self.vllm_server_url}/v1/completions", json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def get_log_probs(self, session: aiohttp.ClientSession, gateway_url: str, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], model_name: str, temperature: float = 0.0) -> Tuple[List[float], List[int]]:
        tasks = []
        prompts_data = []
        
        for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
            full_prompt, prefix_len, comp_len = self._build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
            prompts_data.append((prefix_len, comp_len))
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "max_tokens": 0,
                "temperature": temperature,
                "echo": True,
                "logprobs": 1,
            }
            
            task = self._make_vllm_request(session, self.vllm_server_url, payload)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"GATHER ERROR: {e}")
            raise
        
        # Process results
        log_probs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"TASK {i} FAILED: {result}")
                log_probs.append(0.0)
            else:
                prefix_len, comp_len = prompts_data[i]
                try:
                    log_prob = self.sum_completion_logprobs(result, prefix_len, comp_len)
                    log_probs.append(log_prob)
                except Exception as e:
                    print(f"Parse error for task {i}: {e}")
                    log_probs.append(0.0)
        
        token_counts = [comp_len for _, comp_len in prompts_data]
        
        return log_probs, token_counts

    async def compute_rewards(self, user_data: List[Dict[str, Any]], 
                            user_id: int,
                            save_dir: str,
                            split: str = "train",
                            batch_size: int = 20) -> torch.Tensor:
        """
        Computes raw log probability scores and counts for base and attribute prompts.
        Saves all matrices to .pt file.
        
        Args:
            user_data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            user_id: user id
            save_dir: directory to save the reward matrices
            batch_size: number of samples to process in each batch
        """

        import logging
        
        # Validate components
        if self.tokenizer is None or self.vllm_server_url is None or self.model_name is None or self.base_prompt is None:
            raise RuntimeError("Collector not properly initialized")
        B = len(user_data)
        d_attrs = len(self.attribute_prompts)

        for batch_start in range(0, B, batch_size):
            batch_end = min(batch_start + batch_size, B)
            batch_data = user_data[batch_start:batch_end]
            
            prompts: List[str] = [example["prompt"] for example in batch_data]
            chosen: List[str] = [example["chosen"] for example in batch_data]
            rejected: List[str] = [example["rejected"] for example in batch_data]

            payloads: List[Dict[str, Any]] = []
            # (sample index, attribute index, completion length, 
            #(0, 1, 2, 3) base_chosen (0), attr_chosen (1), base_rejected (2), attr_rejected (3))
            metas: List[Tuple[int, int, int, int]] = []
            prefix_and_len: List[Tuple[int, int]] = []

            for i in range(len(prompts)):
                # Base prompt for chosen outputs
                full_prompt, prefix_len, comp_len = self._build_full_prompt(
                    self.tokenizer, self.base_prompt, prompts[i], chosen[i]
                )
                payloads.append({
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "echo": True,
                    "logprobs": 1,
                })
                metas.append((i, -1, comp_len, 0))
                prefix_and_len.append((prefix_len, comp_len))

                # Attribute prompt for chosen outputs
                for a_idx, a_sys in enumerate(self.attribute_prompts):
                    full_prompt, prefix_len, comp_len = self._build_full_prompt(
                        self.tokenizer, a_sys, prompts[i], chosen[i]
                    )
                    payloads.append({
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "max_tokens": 0,
                        "temperature": 0.0,
                        "echo": True,
                        "logprobs": 1,
                    })
                    metas.append((i, a_idx, comp_len, 1))
                    prefix_and_len.append((prefix_len, comp_len))
                
                # Base prompt for rejected outputs
                full_prompt, prefix_len, comp_len = self._build_full_prompt(
                    self.tokenizer, self.base_prompt, prompts[i], rejected[i]
                )
                payloads.append({
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "echo": True,
                    "logprobs": 1,
                })
                metas.append((i, -1, comp_len, 2))
                prefix_and_len.append((prefix_len, comp_len))

                # Attribute prompt for rejected outputs
                for a_idx, a_sys in enumerate(self.attribute_prompts):
                    full_prompt, prefix_len, comp_len = self._build_full_prompt(
                        self.tokenizer, a_sys, prompts[i], rejected[i]
                    )
                    payloads.append({
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "max_tokens": 0,
                        "temperature": 0.8,
                        "echo": True,
                        "logprobs": 1,
                    })
                    metas.append((i, a_idx, comp_len, 3))
                    prefix_and_len.append((prefix_len, comp_len))
            
            total_requests = len(payloads)
            logging.info(f"VLLM REQUESTS: Starting {total_requests} requests via Gateway")
            logging.info(f"Using timeout: {self.request_timeout}s, max_retries: {self.max_retries}, max_concurrent: {self.max_concurrent_requests}")
            start_time = time.time()

            raw_results = await self._post_with_retries_async("/v1/completions", payloads, use_tqdm=True)

            elapsed = time.time() - start_time
            logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")

            base_scores_chosen = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            base_counts_chosen = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            attr_scores_chosen = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")
            attr_counts_chosen = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")

            base_scores_rejected = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            base_counts_rejected = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            attr_scores_rejected = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")
            attr_counts_rejected = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")

            for idx, res in enumerate(raw_results):
                i, a_idx, comp_len, group = metas[idx]
                prefix_len, comp_len_eff = prefix_and_len[idx]

                if res is None or isinstance(res, Exception):
                    error_msg = str(res) if res is not None else "Request failed after retries"
                    logging.warning(f"Score failed for sample={i}, attr={a_idx}, group={group}: {error_msg}")
                    continue

                try:
                    s = self.sum_completion_logprobs(res, prefix_len, comp_len_eff)
                    if group == 0:
                        base_scores_chosen[i] += s
                        base_counts_chosen[i] += max(1, comp_len_eff)
                    elif group == 1:
                        attr_scores_chosen[i, a_idx] += s
                        attr_counts_chosen[i, a_idx] += max(1, comp_len_eff)
                    elif group == 2:
                        base_scores_rejected[i] += s
                        base_counts_rejected[i] += max(1, comp_len_eff)
                    elif group == 3:
                        attr_scores_rejected[i, a_idx] += s
                        attr_counts_rejected[i, a_idx] += max(1, comp_len_eff)
                except Exception as e:
                    logging.warning(f"Parse failed for sample={i}, attr={a_idx}: {e}")

            results_dict = {
                'base_scores_chosen': base_scores_chosen,
                'base_counts_chosen': base_counts_chosen,
                'attr_scores_chosen': attr_scores_chosen,
                'attr_counts_chosen': attr_counts_chosen,
                'base_scores_rejected': base_scores_rejected,
                'base_counts_rejected': base_counts_rejected,
                'attr_scores_rejected': attr_scores_rejected,
                'attr_counts_rejected': attr_counts_rejected,
            }
            
            # Load existing data or initialize new structure
            file_path = pathlib.Path(f"{save_dir}/{split}/user{user_id}.pt")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                formed_data = torch.load(file_path)
                
                for key in results_dict:
                    if key != 'metadata':
                        formed_data[key] = torch.cat([formed_data[key], results_dict[key]])
            else:
                formed_data = results_dict

            torch.save(formed_data, file_path)

    async def compute_rewards_multi_turn(self, user_data: List[Dict[str, Any]], 
                            user_id: int,
                            save_dir: str,
                            split: str = "train",
                            batch_size: int = 20) -> torch.Tensor:
        """
        Computes raw log probability scores and counts for base and attribute prompts.
        Saves all matrices to .pt file.
        
        Args:
            user_data: List of dicts with 'prompt', 'chosen', 'rejected' keys (multi-turn)
            user_id: user id
            save_dir: directory to save the reward matrices
            batch_size: number of samples to process in each batch
        """

        import logging
        
        # Validate components
        if self.tokenizer is None or self.vllm_server_url is None or self.model_name is None or self.base_prompt is None:
            raise RuntimeError("Collector not properly initialized")
        B = len(user_data)
        d_attrs = len(self.attribute_prompts)

        for batch_start in range(0, B, batch_size):
            batch_end = min(batch_start + batch_size, B)
            batch_data = user_data[batch_start:batch_end]
            
            prompts: List[str] = [example["prompt"] for example in batch_data]
            chosen: List[str] = [example["chosen"] for example in batch_data]
            rejected: List[str] = [example["rejected"] for example in batch_data]

            payloads: List[Dict[str, Any]] = []
            # (sample index, attribute index, completion length, 
            #(0, 1, 2, 3) base_chosen (0), attr_chosen (1), base_rejected (2), attr_rejected (3))
            metas: List[Tuple[int, int, int, int]] = []
            prefix_and_len: List[Tuple[int, int]] = []

            for i in range(len(prompts)):
                # Base prompt for chosen outputs
                full_prompt, prefix_len, comp_len = self._build_full_prompt_multi_turn(
                    self.tokenizer, self.base_prompt, prompts[i], chosen[i]
                )
                payloads.append({
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "echo": True,
                    "logprobs": 1,
                })
                metas.append((i, -1, comp_len, 0))
                prefix_and_len.append((prefix_len, comp_len))

                # Attribute prompt for chosen outputs
                for a_idx, a_sys in enumerate(self.attribute_prompts):
                    full_prompt, prefix_len, comp_len = self._build_full_prompt_multi_turn(
                        self.tokenizer, a_sys, prompts[i], chosen[i]
                    )
                    payloads.append({
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "max_tokens": 0,
                        "temperature": 0.0,
                        "echo": True,
                        "logprobs": 1,
                    })
                    metas.append((i, a_idx, comp_len, 1))
                    prefix_and_len.append((prefix_len, comp_len))
                
                # Base prompt for rejected outputs
                full_prompt, prefix_len, comp_len = self._build_full_prompt_multi_turn(
                    self.tokenizer, self.base_prompt, prompts[i], rejected[i]
                )
                payloads.append({
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "echo": True,
                    "logprobs": 1,
                })
                metas.append((i, -1, comp_len, 2))
                prefix_and_len.append((prefix_len, comp_len))

                # Attribute prompt for rejected outputs
                for a_idx, a_sys in enumerate(self.attribute_prompts):
                    full_prompt, prefix_len, comp_len = self._build_full_prompt_multi_turn(
                        self.tokenizer, a_sys, prompts[i], rejected[i]
                    )
                    payloads.append({
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "max_tokens": 0,
                        "temperature": 0.8,
                        "echo": True,
                        "logprobs": 1,
                    })
                    metas.append((i, a_idx, comp_len, 3))
                    prefix_and_len.append((prefix_len, comp_len))
            
            total_requests = len(payloads)
            logging.info(f"VLLM REQUESTS: Starting {total_requests} requests via Gateway")
            logging.info(f"Using timeout: {self.request_timeout}s, max_retries: {self.max_retries}, max_concurrent: {self.max_concurrent_requests}")
            start_time = time.time()

            raw_results = await self._post_with_retries_async("/v1/completions", payloads, use_tqdm=True)

            elapsed = time.time() - start_time
            logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")

            base_scores_chosen = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            base_counts_chosen = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            attr_scores_chosen = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")
            attr_counts_chosen = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")

            base_scores_rejected = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            base_counts_rejected = torch.zeros(len(prompts), dtype=torch.float32, device="cpu")
            attr_scores_rejected = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")
            attr_counts_rejected = torch.zeros(len(prompts), d_attrs, dtype=torch.float32, device="cpu")

            for idx, res in enumerate(raw_results):
                i, a_idx, comp_len, group = metas[idx]
                prefix_len, comp_len_eff = prefix_and_len[idx]

                if res is None or isinstance(res, Exception):
                    error_msg = str(res) if res is not None else "Request failed after retries"
                    logging.warning(f"Score failed for sample={i}, attr={a_idx}, group={group}: {error_msg}")
                    continue

                try:
                    s = self.sum_completion_logprobs(res, prefix_len, comp_len_eff)
                    if group == 0:
                        base_scores_chosen[i] += s
                        base_counts_chosen[i] += max(1, comp_len_eff)
                    elif group == 1:
                        attr_scores_chosen[i, a_idx] += s
                        attr_counts_chosen[i, a_idx] += max(1, comp_len_eff)
                    elif group == 2:
                        base_scores_rejected[i] += s
                        base_counts_rejected[i] += max(1, comp_len_eff)
                    elif group == 3:
                        attr_scores_rejected[i, a_idx] += s
                        attr_counts_rejected[i, a_idx] += max(1, comp_len_eff)
                except Exception as e:
                    logging.warning(f"Parse failed for sample={i}, attr={a_idx}: {e}")

            results_dict = {
                'base_scores_chosen': base_scores_chosen,
                'base_counts_chosen': base_counts_chosen,
                'attr_scores_chosen': attr_scores_chosen,
                'attr_counts_chosen': attr_counts_chosen,
                'base_scores_rejected': base_scores_rejected,
                'base_counts_rejected': base_counts_rejected,
                'attr_scores_rejected': attr_scores_rejected,
                'attr_counts_rejected': attr_counts_rejected,
            }
            
            # Load existing data or initialize new structure
            file_path = pathlib.Path(f"{save_dir}/{split}/user{user_id}.pt")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                formed_data = torch.load(file_path)
                
                for key in results_dict:
                    if key != 'metadata':
                        formed_data[key] = torch.cat([formed_data[key], results_dict[key]])
            else:
                formed_data = results_dict

            torch.save(formed_data, file_path)

    def l1_solve(self, d_mean, l1_lambda, std=None):
        """
        Closed-form solution to: maximize d^T p - lambda * ||p||_1  s.t. ||p||_2 <= 1
        """
        d = np.asarray(d_mean, dtype=float)
        # soft-threshold
        z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
        norm = np.linalg.norm(z, ord=2)
        if norm == 0.0:
            return np.zeros_like(d)
        if std is None:
            return z / norm
        else:
            return z / (norm * std)

    async def approximate(self, data: List[Tuple[str, str, str]], s0: str, s_list: List[str], l1_lambda: float = 0.01) -> np.ndarray:
        """
        Async version using VLLM gateway for the approximate function
        
        Args:
            gateway_url: URL of the VLLM-compatible gateway
            data: list of (question, y_w, y_l) tuples
            tokenizer: tokenizer
            model_name: model identifier
            s0: base system prompt
            s_list: list of attribute system prompts
            l1_lambda: L1 regularization parameter
        
        Returns:
            p vector (numpy array)
        """
        
        m, k = len(data), len(s_list)
        questions, yw_list, yl_list = zip(*data)
        
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Compute base probabilities
            print("Computing base probabilities...")
            pi_yw_base, cnt_yw_base = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [s0]*m, questions, yw_list, self.model_name)
            pi_yl_base, cnt_yl_base = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [s0]*m, questions, yl_list, self.model_name)

            # Convert to tensors
            pi_yw_base = torch.tensor(pi_yw_base, dtype=torch.float32)
            cnt_yw_base = torch.tensor(cnt_yw_base, dtype=torch.float32)
            pi_yl_base = torch.tensor(pi_yl_base, dtype=torch.float32)
            cnt_yl_base = torch.tensor(cnt_yl_base, dtype=torch.float32)

            # Safe average log-probs
            eps = 1e-12
            yw_base_avg = pi_yw_base / torch.clamp(cnt_yw_base, min=eps)
            yl_base_avg = pi_yl_base / torch.clamp(cnt_yl_base, min=eps)

            # Build X matrix
            X = torch.zeros((m, k), dtype=torch.float32)
            
            for j, system in enumerate(s_list):
                print(f"Processing attribute {j+1}/{k}: {system[:50]}...")
                
                pi_yw_attr, cnt_yw_attr = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [system]*m, questions, yw_list, self.model_name)
                pi_yl_attr, cnt_yl_attr = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [system]*m, questions, yl_list, self.model_name)
                
                pi_yw_attr = torch.tensor(pi_yw_attr, dtype=torch.float32)
                cnt_yw_attr = torch.tensor(cnt_yw_attr, dtype=torch.float32)
                pi_yl_attr = torch.tensor(pi_yl_attr, dtype=torch.float32)
                cnt_yl_attr = torch.tensor(cnt_yl_attr, dtype=torch.float32)
                
                yw_attr_avg = pi_yw_attr / torch.clamp(cnt_yw_attr, min=eps)
                yl_attr_avg = pi_yl_attr / torch.clamp(cnt_yl_attr, min=eps)
                
                X[:, j] = (yw_attr_avg - yw_base_avg) - (yl_attr_avg - yl_base_avg)
        
        col_std = X.std(dim=0).clamp_min(1e-8)
        d = (X / col_std).mean(dim=0).detach().cpu().numpy()
        
        p = self.l1_solve(d, l1_lambda, std=col_std.detach().cpu().numpy())
        
        return p

    async def _post_with_retries_async(self, endpoint: str, payload: List[Dict[str, Any]], use_tqdm: bool = False) -> List[Dict[str, Any]]:
        """
        Submit requests with retry logic and connection pooling.
        Args:
            endpoint: API endpoint to call
            payload: List of payloads to send
            use_tqdm: Whether to show progress bar
        Returns:
            List of responses
        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Create ONE session with connection pooling for ALL requests
        # Since server is a single host, set limits based on max_concurrent_requests
        # Connection reuse means actual connections needed < max_concurrent_requests
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,  # Total connection pool size
            limit_per_host=self.max_concurrent_requests,  # Max connections to server
            ttl_dns_cache=600,
            enable_cleanup_closed=True
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            async def _make_request_with_retries(p):
                for attempt in range(self.max_retries):
                    try:
                        async with session.post(
                            f"{self.vllm_server_url}{endpoint}",
                            json=p,
                            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            if isinstance(data, dict):
                                data = [data]
                            return data
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                        await asyncio.sleep(2)
            # Limit concurrent requests using a semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            async def limited_request(p):
                async with semaphore:
                    return await _make_request_with_retries(p)
            tasks = [limited_request(p) for p in payload]
            if use_tqdm:
                from tqdm import tqdm
                # Create a wrapper to update progress bar
                completed_count = [0]
                pbar = tqdm(total=len(tasks), desc="Processing")
                async def tracked_task(task):
                    result = await task
                    completed_count[0] += 1
                    pbar.update(1)
                    return result
                results = await asyncio.gather(*[tracked_task(task) for task in tasks])
                pbar.close()
            else:
                results = await asyncio.gather(*tasks)
        # Flatten results
        flattened_results = []
        for result in results:
            flattened_results.extend(result)
        return flattened_results

    async def evaluate_accuracy(self, test_data: List[Dict[str, str]], p: np.ndarray) -> float:
        """
        Evaluate preference pair accuracy on test data using the learned p vector and VLLM gateway
        
        Args:
            gateway_url: URL of the VLLM-compatible gateway
            test_data: list of preference pairs with 'prompt', 'chosen', 'rejected'
            p: learned drift vector
            tokenizer: tokenizer
            model_name: model identifier
            base_prompt: base system prompt
            attribute_prompts: list of attribute prompts
        
        Returns:
            accuracy (float)
        """
        
        n = len(test_data)
        prompts = [item['prompt'] for item in test_data]
        chosen = [item['chosen'] for item in test_data]
        rejected = [item['rejected'] for item in test_data]
        
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("Computing base log probabilities for test data...")
            chosen_base_probs, chosen_base_counts = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [self.base_prompt]*n, prompts, chosen, self.model_name)
            rejected_base_probs, rejected_base_counts = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [self.base_prompt]*n, prompts, rejected, self.model_name)
            
            drift_scores = torch.zeros(n, dtype=torch.float32)
            
            for i, attr_prompt in enumerate(self.attribute_prompts):
                if p[i] == 0:
                    continue
                    
                print(f"Processing test attribute {i+1}/{len(self.attribute_prompts)}: p={p[i]:.4f}")
                
                chosen_attr_probs, chosen_attr_counts = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [attr_prompt]*n, prompts, chosen, self.model_name)
                rejected_attr_probs, rejected_attr_counts = await self.get_log_probs(session, self.vllm_server_url, self.tokenizer, [attr_prompt]*n, prompts, rejected, self.model_name)
                
                chosen_attr_avg = torch.tensor(chosen_attr_probs, dtype=torch.float32) / torch.tensor(chosen_attr_counts, dtype=torch.float32)
                rejected_attr_avg = torch.tensor(rejected_attr_probs, dtype=torch.float32) / torch.tensor(rejected_attr_counts, dtype=torch.float32)
                chosen_base_avg = torch.tensor(chosen_base_probs, dtype=torch.float32) / torch.tensor(chosen_base_counts, dtype=torch.float32)
                rejected_base_avg = torch.tensor(rejected_base_probs, dtype=torch.float32) / torch.tensor(rejected_base_counts, dtype=torch.float32)
                
                attribute_drift = p[i] * ((chosen_attr_avg - chosen_base_avg) - (rejected_attr_avg - rejected_base_avg))
                drift_scores += attribute_drift
        
        correct = (drift_scores > 0).sum().item()
        accuracy = correct / n
        
        return accuracy