import asyncio
import aiohttp
import os
import time
from typing import Tuple, List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/completions")
MODEL_ID  = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
CONCURRENCY = int(os.getenv("VLLM_CONCURRENCY", "256"))
MAX_RETRIES = int(os.getenv("VLLM_MAX_RETRIES", "3"))

def build_full_prompt(tokenizer, sys_prompt: str, user_prompt: str, completion: str) -> Tuple[str, int, int]:
    """Return: full_text (prompt+completion), n_prefix_tokens, completion_len_tokens"""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt.strip()},
         {"role": "user",   "content": user_prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    return prompt_text + completion, len(prompt_ids), len(comp_ids)

def sum_completion_logprobs(resp_json, n_prefix: int, comp_len: int) -> float:
    lp = resp_json["choices"][0]["logprobs"]["token_logprobs"]
    end = min(len(lp), n_prefix + comp_len)
    seg = [x for x in lp[n_prefix:end] if x is not None]
    return float(sum(seg))

async def fetch_sum_lp(session: aiohttp.ClientSession, prompt: str, n_prefix: int, comp_len: int, vllm_url: str = None, model_id: str = None, max_retries: int = None) -> float:
    payload = {
        "model": model_id or MODEL_ID,
        "prompt": prompt,
        "echo": True,
        "logprobs": 1,
        "max_tokens": 0,
        "temperature": 0.0,
    }
    url = vllm_url or VLLM_URL
    max_retries = max_retries or MAX_RETRIES
    
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                return sum_completion_logprobs(data, n_prefix, comp_len)
        except (aiohttp.ClientConnectionResetError, aiohttp.ClientOSError, aiohttp.ClientConnectorError) as e:
            if attempt == max_retries - 1:
                print(f"VLLM REQUEST FAILED after {max_retries} attempts: {e}")
                raise
            print(f"VLLM CONNECTION ERROR (attempt {attempt + 1}/{max_retries}): {e} - retrying...")
            await asyncio.sleep(0.5 * (2 ** attempt))  # exponential backoff
        except Exception as e:
            print(f"VLLM REQUEST ERROR: {e}")
            raise

async def get_log_probs_async(session: aiohttp.ClientSession, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], vllm_url: str = None, model_id: str = None) -> Tuple[List[float], List[int]]:
    """
    Async version of get_log_probs using aiohttp
    """
    tasks = []
    prompts_data = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        full_prompt, n_prefix, comp_len = build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
        prompts_data.append((n_prefix, comp_len))
        tasks.append(fetch_sum_lp(session, full_prompt, n_prefix, comp_len, vllm_url, model_id))
    
    try:
        log_probs = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"GATHER ERROR: {e}")
        raise
    
    # Check for exceptions in results
    for i, result in enumerate(log_probs):
        if isinstance(result, Exception):
            print(f"TASK {i} FAILED: {result}")
            raise result
    
    token_counts = [comp_len for _, comp_len in prompts_data]
    
    return log_probs, token_counts

async def compute_drift_rewards(session: aiohttp.ClientSession, tokenizer, prompts: List[str], outputs: List[str], 
                               base_prompt: str, attribute_prompts: List[str], vllm_url: str = None, model_id: str = None, device = None) -> torch.Tensor:
    """
    Compute drift rewards: attr_avg_logprob - base_avg_logprob (shape [B, d])
    OPTIMIZED VERSION: Fires all (d+1)*B requests concurrently in one wave
    """
    import torch
    
    B = len(outputs)
    d = len(attribute_prompts)
    if B == 0:
        return torch.zeros(0, d, device=device)

    # Build ALL requests at once: base + all attributes
    all_system_prompts = []
    all_user_prompts = []
    all_completion_texts = []
    request_map = []  # Track which request corresponds to which (type, attr_idx, sample_idx)
    
    # Base prompts for all samples
    for i in range(B):
        all_system_prompts.append(base_prompt)
        all_user_prompts.append(prompts[i])
        all_completion_texts.append(outputs[i])
        request_map.append(("base", 0, i))
    
    # Attribute prompts for all samples
    for attr_idx in range(d):
        for i in range(B):
            all_system_prompts.append(attribute_prompts[attr_idx])
            all_user_prompts.append(prompts[i])
            all_completion_texts.append(outputs[i])
            request_map.append(("attr", attr_idx, i))
    
    # Process requests in manageable batches to avoid overwhelming the system
    total_requests = len(all_system_prompts)
    # Use configurable batch size from environment or default to 512
    default_batch_size = int(os.getenv("REQUEST_BATCH_SIZE", "512"))
    batch_size = min(default_batch_size, total_requests)  # Process in manageable chunks
    
    print(f"Processing {total_requests} requests in batches of {batch_size} ({B} samples × {d+1} prompts)...")
    start_time = time.time()
    
    all_probs = []
    all_counts = []
    
    for i in range(0, total_requests, batch_size):
        end_i = min(i + batch_size, total_requests)
        batch_system = all_system_prompts[i:end_i]
        batch_user = all_user_prompts[i:end_i]
        batch_completion = all_completion_texts[i:end_i]
        
        print(f"  Processing batch {i//batch_size + 1}/{(total_requests + batch_size - 1)//batch_size} ({end_i - i} requests)...")
        
        try:
            batch_probs, batch_counts = await get_log_probs_async(
                session, tokenizer, batch_system, batch_user, batch_completion, vllm_url, model_id
            )
        except Exception as e:
            print(f"BATCH PROCESSING ERROR (batch {i//batch_size + 1}): {e}")
            # For partial failure, we could implement fallback logic here
            raise
        
        all_probs.extend(batch_probs)
        all_counts.extend(batch_counts)
    
    elapsed = time.time() - start_time
    print(f"Completed {total_requests} requests in {elapsed:.2f}s ({total_requests/elapsed:.1f} req/sec)")
    
    # Reconstruct base scores and attribute scores
    base_scores = torch.zeros(B, device=device)
    attr_scores = torch.zeros(B, d, device=device)
    
    for i, (request_type, attr_idx, sample_idx) in enumerate(request_map):
        score = all_probs[i] / max(all_counts[i], 1)  # Avoid division by zero
        
        if request_type == "base":
            base_scores[sample_idx] = score
        else:  # attr
            attr_scores[sample_idx, attr_idx] = score
    
    # Compute drift: attr_score - base_score for each attribute
    reward_matrix = attr_scores - base_scores.unsqueeze(1)  # Broadcast base_scores across attributes
    
    return reward_matrix

async def compute_drift_rewards_old(session: aiohttp.ClientSession, tokenizer, prompts: List[str], outputs: List[str], 
                               base_prompt: str, attribute_prompts: List[str], vllm_url: str = None, model_id: str = None, device = None) -> torch.Tensor:
    """
    OLD SEQUENTIAL VERSION: Compute drift rewards sequentially (kept for reference)
    """
    import torch
    
    B = len(outputs)
    d = len(attribute_prompts)
    if B == 0:
        return torch.zeros(0, d, device=device)

    # Get base log probabilities
    base_probs, base_counts = await get_log_probs_async(session, tokenizer, [base_prompt] * B, prompts, outputs, vllm_url, model_id)
    base_scores = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Build reward matrix
    reward_matrix = torch.zeros(B, d, device=device)
    
    # Compute drift scores for each attribute
    for attr_idx in range(d):
        attr_prompt = attribute_prompts[attr_idx]
        attr_probs, attr_counts = await get_log_probs_async(session, tokenizer, [attr_prompt] * B, prompts, outputs, vllm_url, model_id)
        attr_scores = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        reward_matrix[:, attr_idx] = attr_scores - base_scores
    
    return reward_matrix

def l1_solve(d_mean, l1_lambda, std=None):
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

async def approximate_async(data: List[Tuple[str, str, str]], tokenizer, s0: str, s_list: List[str], l1_lambda: float = 0.01) -> np.ndarray:
    """
    Async version of the approximate function from drift.py
    
    Args:
        data: list of (question, y_w, y_l) tuples
        tokenizer: tokenizer
        s0: base system prompt
        s_list: list of attribute system prompts
        l1_lambda: L1 regularization parameter
    
    Returns:
        p vector (numpy array)
    """
    m, k = len(data), len(s_list)
    questions, yw_list, yl_list = zip(*data)
    
    # Set up async session
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Compute base probabilities
        print("Computing base probabilities...")
        pi_yw_base, cnt_yw_base = await get_log_probs_async(session, tokenizer, [s0]*m, questions, yw_list)
        pi_yl_base, cnt_yl_base = await get_log_probs_async(session, tokenizer, [s0]*m, questions, yl_list)
        
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
            
            pi_yw_attr, cnt_yw_attr = await get_log_probs_async(session, tokenizer, [system]*m, questions, yw_list)
            pi_yl_attr, cnt_yl_attr = await get_log_probs_async(session, tokenizer, [system]*m, questions, yl_list)
            
            pi_yw_attr = torch.tensor(pi_yw_attr, dtype=torch.float32)
            cnt_yw_attr = torch.tensor(cnt_yw_attr, dtype=torch.float32)
            pi_yl_attr = torch.tensor(pi_yl_attr, dtype=torch.float32)
            cnt_yl_attr = torch.tensor(cnt_yl_attr, dtype=torch.float32)
            
            yw_attr_avg = pi_yw_attr / torch.clamp(cnt_yw_attr, min=eps)
            yl_attr_avg = pi_yl_attr / torch.clamp(cnt_yl_attr, min=eps)
            
            # Column j: (yw_attr - yw_base) - (yl_attr - yl_base)
            X[:, j] = (yw_attr_avg - yw_base_avg) - (yl_attr_avg - yl_base_avg)
    
    # Compute drift direction
    col_std = X.std(dim=0).clamp_min(1e-8)
    d = (X / col_std).mean(dim=0).detach().cpu().numpy()
    
    # Solve for p vector
    p = l1_solve(d, l1_lambda, std=col_std.detach().cpu().numpy())
    
    return p

async def evaluate_accuracy_async(test_data: List[Dict[str, str]], p: np.ndarray, tokenizer, base_prompt: str, attribute_prompts: List[str]) -> float:
    """
    Evaluate preference pair accuracy on test data using the learned p vector
    
    Args:
        test_data: list of preference pairs with 'prompt', 'chosen', 'rejected'
        p: learned drift vector
        tokenizer: tokenizer
        base_prompt: base system prompt
        attribute_prompts: list of attribute prompts
    
    Returns:
        accuracy (float)
    """
    n = len(test_data)
    prompts = [item['prompt'] for item in test_data]
    chosen = [item['chosen'] for item in test_data]
    rejected = [item['rejected'] for item in test_data]
    
    # Set up async session
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Get base log probabilities
        print("Computing base log probabilities for test data...")
        chosen_base_probs, chosen_base_counts = await get_log_probs_async(session, tokenizer, [base_prompt]*n, prompts, chosen)
        rejected_base_probs, rejected_base_counts = await get_log_probs_async(session, tokenizer, [base_prompt]*n, prompts, rejected)
        
        # Initialize drift scores
        drift_scores = torch.zeros(n, dtype=torch.float32)
        
        # Process each attribute
        for i, attr_prompt in enumerate(attribute_prompts):
            if p[i] == 0:
                continue
                
            print(f"Processing test attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
            
            chosen_attr_probs, chosen_attr_counts = await get_log_probs_async(session, tokenizer, [attr_prompt]*n, prompts, chosen)
            rejected_attr_probs, rejected_attr_counts = await get_log_probs_async(session, tokenizer, [attr_prompt]*n, prompts, rejected)
            
            # Convert to tensors and compute averages
            chosen_attr_avg = torch.tensor(chosen_attr_probs, dtype=torch.float32) / torch.tensor(chosen_attr_counts, dtype=torch.float32)
            rejected_attr_avg = torch.tensor(rejected_attr_probs, dtype=torch.float32) / torch.tensor(rejected_attr_counts, dtype=torch.float32)
            chosen_base_avg = torch.tensor(chosen_base_probs, dtype=torch.float32) / torch.tensor(chosen_base_counts, dtype=torch.float32)
            rejected_base_avg = torch.tensor(rejected_base_probs, dtype=torch.float32) / torch.tensor(rejected_base_counts, dtype=torch.float32)
            
            # Compute drift contribution: p[i] * ((chosen_attr - chosen_base) - (rejected_attr - rejected_base))
            attribute_drift = p[i] * ((chosen_attr_avg - chosen_base_avg) - (rejected_attr_avg - rejected_base_avg))
            drift_scores += attribute_drift
    
    # Count correct predictions (positive drift means chosen > rejected)
    correct = (drift_scores > 0).sum().item()
    accuracy = correct / n
    
    return accuracy