import asyncio
import aiohttp
import os
import time
from typing import Tuple, List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import LogitsProcessor

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1/completions")
MODEL_ID  = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
CONCURRENCY = int(os.getenv("VLLM_CONCURRENCY", "256"))
MAX_RETRIES = int(os.getenv("VLLM_MAX_RETRIES", "3"))

def build_full_prompt(tokenizer, sys_prompt: str, user_prompt: str, completion: str) -> Tuple[str, int, int]:
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

def sum_completion_logprobs(resp_json, prefix_len: int, comp_len: int) -> float:
    lp = resp_json["choices"][0]["logprobs"]["token_logprobs"]
    end = min(len(lp), prefix_len + comp_len)
    seg = [x for x in lp[prefix_len:end] if x is not None]
    return float(sum(seg))

async def fetch_sum_lp(session: aiohttp.ClientSession, prompt: str, prefix_len: int, comp_len: int, vllm_url: str = None, model_id: str = None, max_retries: int = None) -> float:
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
                return sum_completion_logprobs(data, prefix_len, comp_len)
        except (aiohttp.ClientConnectionResetError, aiohttp.ClientOSError, aiohttp.ClientConnectorError) as e:
            if attempt == max_retries - 1:
                print(f"VLLM REQUEST FAILED after {max_retries} attempts: {e}")
                raise
            print(f"VLLM CONNECTION ERROR (attempt {attempt + 1}/{max_retries}): {e} - retrying...")
            await asyncio.sleep(0.5 * (2 ** attempt))  # exponential backoff
        except Exception as e:
            print(f"VLLM REQUEST ERROR: {e}")
            raise

async def get_log_probs_async_registry(http_client, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], model_name: str) -> Tuple[List[float], List[int]]:
    """
    Registry-based async version of get_log_probs using RegistryHTTPClient
    """
    tasks = []
    prompts_data = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        full_prompt, prefix_len, comp_len = build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
        prompts_data.append((prefix_len, comp_len))
        
        # Create payload for registry client
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        }
        
        # Use registry client to make request
        task = http_client.request_with_rotation("v1/completions", payload)
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
            log_probs.append(0.0)  # Default value for failed requests
        else:
            response, _ = result  # Unpack (response, server_idx)
            prefix_len, comp_len = prompts_data[i]
            try:
                log_prob = sum_completion_logprobs(response, prefix_len, comp_len)
                log_probs.append(log_prob)
            except Exception as e:
                print(f"Parse error for task {i}: {e}")
                log_probs.append(0.0)
    
    token_counts = [comp_len for _, comp_len in prompts_data]
    
    return log_probs, token_counts

async def get_log_probs_async(session: aiohttp.ClientSession, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], vllm_url: str = None, model_id: str = None) -> Tuple[List[float], List[int]]:
    """
    Async version of get_log_probs using aiohttp
    """
    tasks = []
    prompts_data = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        full_prompt, prefix_len, comp_len = build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
        prompts_data.append((prefix_len, comp_len))
        tasks.append(fetch_sum_lp(session, full_prompt, prefix_len, comp_len, vllm_url, model_id))
    
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



async def compute_drift_rewards(registry_client, tokenizer, prompts: List[str], outputs: List[str], 
                               base_prompt: str, attribute_prompts: List[str], model_name: str = None, device = None) -> torch.Tensor:
    """
    Compute drift rewards using LiteRegistry: attr_avg_logprob - base_avg_logprob (shape [B, d])
    OPTIMIZED VERSION: Uses registry for efficient server management and load balancing
    """
    import torch
    from literegistry.literegistry.http import RegistryHTTPClient
    
    B = len(outputs)
    d = len(attribute_prompts)
    if B == 0:
        return torch.zeros(0, d, device=device)

    # Build ALL requests at once: base + all attributes
    payloads = []
    request_map = []  # Track which request corresponds to which (type, attr_idx, sample_idx)
    
    # Base prompts for all samples
    for i in range(B):
        full_prompt, prefix_len, comp_len = build_full_prompt(tokenizer, base_prompt, prompts[i], outputs[i])
        payloads.append({
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        })
        request_map.append(("base", 0, i, prefix_len, comp_len))
    
    # Attribute prompts for all samples
    for attr_idx in range(d):
        for i in range(B):
            full_prompt, prefix_len, comp_len = build_full_prompt(tokenizer, attribute_prompts[attr_idx], prompts[i], outputs[i])
            payloads.append({
                "model": model_name,
                "prompt": full_prompt,
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            })
            request_map.append(("attr", attr_idx, i, prefix_len, comp_len))
    
    total_requests = len(payloads)
    print(f"Processing {total_requests} requests via LiteRegistry ({B} samples × {d+1} prompts)...")
    start_time = time.time()
    
    # Check server availability
    available_servers = await registry_client.get_all(model_name)
    if not available_servers:
        raise RuntimeError(f"No servers available for model {model_name}")
    
    # Use RegistryHTTPClient for efficient batch processing
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
    REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
    
    async with RegistryHTTPClient(
        registry=registry_client,
        value=model_name,
        max_parallel_requests=1024,
        timeout=REQUEST_TIMEOUT,
        max_retries=REQUEST_RETRIES
    ) as httpClient:
        print(f"Sending {len(payloads)} requests via LiteRegistry...")
        raw_results = await httpClient.post("v1/completions", payloads, track=True)
        print("LiteRegistry batch processing completed")
    
    elapsed = time.time() - start_time
    print(f"Completed {total_requests} requests in {elapsed:.2f}s ({total_requests/elapsed:.1f} req/sec)")
    
    # Process results and build reward matrix
    base_scores = torch.zeros(B, device=device)
    attr_scores = torch.zeros(B, d, device=device)
    
    for i, (request_type, attr_idx, sample_idx, prefix_len, comp_len) in enumerate(request_map):
        result = raw_results[i]
        
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
            continue
            
        try:
            score = sum_completion_logprobs(result, prefix_len, comp_len)
            
            if request_type == "base":
                base_scores[sample_idx] = score
            else:  # attr
                attr_scores[sample_idx, attr_idx] = score
        except Exception as e:
            print(f"Parse failed for request {i}: {e}")
    
    # Compute drift: attr_score - base_score for each attribute
    reward_matrix = attr_scores - base_scores.unsqueeze(1)
    
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

async def approximate_async_registry(registry_client, data: List[Tuple[str, str, str]], tokenizer, model_name: str, s0: str, s_list: List[str], l1_lambda: float = 0.01) -> np.ndarray:
    """
    Async version using LiteRegistry for the approximate function
    
    Args:
        registry_client: LiteRegistry client for server management
        data: list of (question, y_w, y_l) tuples
        tokenizer: tokenizer
        model_name: model identifier for registry
        s0: base system prompt
        s_list: list of attribute system prompts
        l1_lambda: L1 regularization parameter
    
    Returns:
        p vector (numpy array)
    """
    from literegistry.literegistry.http import RegistryHTTPClient
    
    m, k = len(data), len(s_list)
    questions, yw_list, yl_list = zip(*data)
    
    # Check server availability
    available_servers = await registry_client.get_all(model_name)
    if not available_servers:
        raise RuntimeError(f"No servers available for model {model_name}")
    
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
    REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
    
    async with RegistryHTTPClient(
        registry=registry_client,
        value=model_name,
        max_parallel_requests=1024,
        timeout=REQUEST_TIMEOUT,
        max_retries=REQUEST_RETRIES
    ) as httpClient:
        # Compute base probabilities
        print("Computing base probabilities...")
        pi_yw_base, cnt_yw_base = await get_log_probs_async_registry(httpClient, tokenizer, [s0]*m, questions, yw_list, model_name)
        pi_yl_base, cnt_yl_base = await get_log_probs_async_registry(httpClient, tokenizer, [s0]*m, questions, yl_list, model_name)
        
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
            
            pi_yw_attr, cnt_yw_attr = await get_log_probs_async_registry(httpClient, tokenizer, [system]*m, questions, yw_list, model_name)
            pi_yl_attr, cnt_yl_attr = await get_log_probs_async_registry(httpClient, tokenizer, [system]*m, questions, yl_list, model_name)
            
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

async def evaluate_accuracy_async_registry(registry_client, test_data: List[Dict[str, str]], p: np.ndarray, tokenizer, model_name: str, base_prompt: str, attribute_prompts: List[str]) -> float:
    """
    Evaluate preference pair accuracy on test data using the learned p vector and LiteRegistry
    
    Args:
        registry_client: LiteRegistry client for server management
        test_data: list of preference pairs with 'prompt', 'chosen', 'rejected'
        p: learned drift vector
        tokenizer: tokenizer
        model_name: model identifier for registry
        base_prompt: base system prompt
        attribute_prompts: list of attribute prompts
    
    Returns:
        accuracy (float)
    """
    from literegistry.literegistry.http import RegistryHTTPClient
    
    n = len(test_data)
    prompts = [item['prompt'] for item in test_data]
    chosen = [item['chosen'] for item in test_data]
    rejected = [item['rejected'] for item in test_data]
    
    # Check server availability
    available_servers = await registry_client.get_all(model_name)
    if not available_servers:
        raise RuntimeError(f"No servers available for model {model_name}")
    
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
    REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
    
    async with RegistryHTTPClient(
        registry=registry_client,
        value=model_name,
        max_parallel_requests=1024,
        timeout=REQUEST_TIMEOUT,
        max_retries=REQUEST_RETRIES
    ) as httpClient:
        # Get base log probabilities
        print("Computing base log probabilities for test data...")
        chosen_base_probs, chosen_base_counts = await get_log_probs_async_registry(httpClient, tokenizer, [base_prompt]*n, prompts, chosen, model_name)
        rejected_base_probs, rejected_base_counts = await get_log_probs_async_registry(httpClient, tokenizer, [base_prompt]*n, prompts, rejected, model_name)
        
        # Initialize drift scores
        drift_scores = torch.zeros(n, dtype=torch.float32)
        
        # Process each attribute
        for i, attr_prompt in enumerate(attribute_prompts):
            if p[i] == 0:
                continue
                
            print(f"Processing test attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
            
            chosen_attr_probs, chosen_attr_counts = await get_log_probs_async_registry(httpClient, tokenizer, [attr_prompt]*n, prompts, chosen, model_name)
            rejected_attr_probs, rejected_attr_counts = await get_log_probs_async_registry(httpClient, tokenizer, [attr_prompt]*n, prompts, rejected, model_name)
            
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

class DriftLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        b: float,
        small_model,
        tokenizer,
        base_prompt: str,
        attribute_prompts: list[str],
        weights: list[float],
    ):
        self.b = b
        self.small_model = small_model
        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.weights = weights

    def get_small_logits(self, input_ids, prompt):
        # PROPER IMPLEMENTATION: Reconstruct conversation with different system prompt
        
        # Decode current input_ids back to text
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # This is complex because we need to:
        # 1. Parse the current conversation structure
        # 2. Extract the user message
        # 3. Reconstruct with different system prompt
        # 4. Re-tokenize
        
        # For now, let's try a simpler approach:
        # Extract everything after the last user message and create new context
        
        try:
            # Find the last user input in the tokenized sequence
            # This is a simplified approach - assumes standard chat template structure
            
            # Decode to text and try to extract user content
            if "<|start_header_id|>user<|end_header_id|>" in current_text:
                # Extract user message
                user_start = current_text.rfind("<|start_header_id|>user<|end_header_id|>")
                user_end = current_text.find("<|eot_id|>", user_start)
                if user_end == -1:
                    user_end = len(current_text)
                
                user_content = current_text[user_start + len("<|start_header_id|>user<|end_header_id|>"):user_end].strip()
                
                # Also extract any assistant content that's already been generated
                assistant_start = current_text.rfind("<|start_header_id|>assistant<|end_header_id|>")
                if assistant_start != -1:
                    assistant_content = current_text[assistant_start + len("<|start_header_id|>assistant<|end_header_id|>"):].strip()
                else:
                    assistant_content = ""
                
                # Create new conversation with different system prompt
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ]
                
                # Create the prompt up to assistant response
                new_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Add any existing assistant content
                if assistant_content:
                    new_prompt += assistant_content
                
                # Tokenize the new sequence
                new_input_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids.to(input_ids.device)
                
                # Run through small model
                with torch.no_grad():
                    out = self.small_model(
                        new_input_ids,
                        attention_mask=torch.ones_like(new_input_ids),
                        use_cache=False
                    )
                
                return out.logits[:, -1]
                
        except Exception as e:
            # Fallback: just use the original input_ids if parsing fails
            pass
        
        # Fallback: use original approach if reconstruction fails
        with torch.no_grad():
            out = self.small_model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False
            )
        
        return out.logits[:, -1]

    def __call__(self, input_ids, aligned_logits):
        # Get base model logits
        h0_small = self.get_small_logits(input_ids, self.base_prompt)

        # Compute drift from attribute prompts
        drift = torch.zeros_like(h0_small)
        for w, attr_prompt in zip(self.weights, self.attribute_prompts):
            if w == 0:
                continue
            hi_small = self.get_small_logits(input_ids, attr_prompt)
            drift += w * (hi_small - h0_small)

        # Apply drift to aligned logits
        return aligned_logits + drift / self.b