import asyncio
import os
import time
from typing import Tuple, List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import LogitsProcessor
import aiohttp

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "5"))
REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

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


async def make_vllm_request(session: aiohttp.ClientSession, gateway_url: str, payload: Dict) -> Dict:
    async with session.post(f"{gateway_url}/v1/completions", json=payload) as response:
        response.raise_for_status()
        return await response.json()

async def get_log_probs(session: aiohttp.ClientSession, gateway_url: str, tokenizer, system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], model_name: str, temperature: float = 0.0) -> Tuple[List[float], List[int]]:
    tasks = []
    prompts_data = []
    
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        full_prompt, prefix_len, comp_len = build_full_prompt(tokenizer, sys_prompt, user_prompt, completion)
        prompts_data.append((prefix_len, comp_len))
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": temperature,
            "echo": True,
            "logprobs": 1,
        }
        
        task = make_vllm_request(session, gateway_url, payload)
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
                log_prob = sum_completion_logprobs(result, prefix_len, comp_len)
                log_probs.append(log_prob)
            except Exception as e:
                print(f"Parse error for task {i}: {e}")
                log_probs.append(0.0)
    
    token_counts = [comp_len for _, comp_len in prompts_data]
    
    return log_probs, token_counts

async def compute_rewards(user_data: List[Dict[str, Any]], 
                        user_id: int,
                        attribute_prompts: List[str],
                        base_prompt: str,
                        tokenizer: AutoTokenizer,
                        vllm_server_url: str,
                        model_name: str) -> torch.Tensor:
    """
    Computes raw log probability scores and counts for base and attribute prompts.
    Returns base_scores_chosen tensor for backward compatibility.
    Saves all matrices to timestamped .pt file.
    
    Args:
        user_data: List of dicts with 'prompt', 'chosen', 'rejected' keys
        user_id: user id
    Returns:
        base_scores_chosen tensor (shape [B])
    """

    import logging
    from literegistry import RegistryClient, get_kvstore, RegistryHTTPClient

    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")

    
    # Validate components
    if tokenizer is None or vllm_server_url is None or model_name is None or base_prompt is None:
        raise RuntimeError("Collector not properly initialized")

    prompts: List[str] = [example["prompt"] for example in user_data]
    chosen: List[str] = [example["chosen"] for example in user_data]
    rejected: List[str] = [example["rejected"] for example in user_data]

    B = len(user_data)
    
    d_attrs = len(attribute_prompts)

    payloads: List[Dict[str, Any]] = []
    # (sample index, attribute index, completion length, 
    #(0, 1, 2, 3) base_chosen (0), attr_chosen (1), base_rejected (2), attr_rejected (3))
    metas: List[Tuple[int, int, int, int]] = []
    prefix_and_len: List[Tuple[int, int]] = []

    for i in range(B):
        # Base prompt for chosen outputs
        full_prompt, prefix_len, comp_len = build_full_prompt(
            tokenizer, base_prompt, prompts[i], chosen[i]
        )
        payloads.append({
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        })
        metas.append((i, -1, comp_len, 0))
        prefix_and_len.append((prefix_len, comp_len))

        # Attribute prompt for chosen outputs
        for a_idx, a_sys in enumerate(attribute_prompts):
            full_prompt, prefix_len, comp_len = build_full_prompt(
                tokenizer, a_sys, prompts[i], chosen[i]
            )
            payloads.append({
                "model": model_name,
                "prompt": full_prompt,
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            })
            metas.append((i, a_idx, comp_len, 1))
            prefix_and_len.append((prefix_len, comp_len))
        
        # Base prompt for rejected outputs
        full_prompt, prefix_len, comp_len = build_full_prompt(
            tokenizer, base_prompt, prompts[i], rejected[i]
        )
        payloads.append({
            "model": model_name,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
        })
        metas.append((i, -1, comp_len, 2))
        prefix_and_len.append((prefix_len, comp_len))

        # Attribute prompt for rejected outputs
        for a_idx, a_sys in enumerate(attribute_prompts):
            full_prompt, prefix_len, comp_len = build_full_prompt(
                tokenizer, a_sys, prompts[i], rejected[i]
            )
            payloads.append({
                "model": model_name,
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
    logging.info(f"Using timeout: {REQUEST_TIMEOUT}s, max_retries: {MAX_RETRIES}")
    start_time = time.time()
    
    async with RegistryHTTPClient(client, model_name, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES, max_parallel_requests=REQUEST_BATCH_SIZE) as http_client:
        logging.info(f"Sending {len(payloads)} requests to gateway...")
        
        # Process payloads in smaller batches
        batch_size = REQUEST_BATCH_SIZE
        all_results = []
        
        for batch_start in range(0, len(payloads), batch_size):
            batch_end = min(batch_start + batch_size, len(payloads))
            batch_payloads = payloads[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_start//batch_size + 1}: requests {batch_start}-{batch_end-1}")
            
            batch_results = await http_client.post(
                "v1/completions",
                batch_payloads,
            )
            
            all_results.extend(batch_results)
            logging.info(f"Completed batch {batch_start//batch_size + 1}: {len(batch_results)} requests")
        
        raw_results = all_results
        logging.info("Gateway batch processing completed")

    elapsed = time.time() - start_time
    logging.info(f"VLLM REQUESTS: Completed {total_requests} requests in {elapsed:.1f}s")

    base_scores_chosen = torch.zeros(B, dtype=torch.float32, device="cpu")
    base_counts_chosen = torch.zeros(B, dtype=torch.float32, device="cpu")
    attr_scores_chosen = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")
    attr_counts_chosen = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")

    base_scores_rejected = torch.zeros(B, dtype=torch.float32, device="cpu")
    base_counts_rejected = torch.zeros(B, dtype=torch.float32, device="cpu")
    attr_scores_rejected = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")
    attr_counts_rejected = torch.zeros(B, d_attrs, dtype=torch.float32, device="cpu")

    for idx, res in enumerate(raw_results):
        i, a_idx, comp_len, group = metas[idx]
        prefix_len, comp_len_eff = prefix_and_len[idx]

        if isinstance(res, Exception):
            logging.warning(f"Score failed for sample={i}, attr={a_idx}, group={group}: {type(res).__name__}: {res}")
            continue

        try:
            s = sum_completion_logprobs(res, prefix_len, comp_len_eff)
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
        'metadata': {
            'B': B,
            'd_attrs': d_attrs,
            'total_requests': total_requests,
            'processing_time': elapsed
        }
    }
    
    filename = f"rewards_persona_testing/user{user_id}.pt"
    torch.save(results_dict, filename)
    logging.info(f"Saved reward matrices to {filename}")


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

async def approximate(gateway_url: str, data: List[Tuple[str, str, str]], tokenizer, model_name: str, s0: str, s_list: List[str], l1_lambda: float = 0.01) -> np.ndarray:
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
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Compute base probabilities
        print("Computing base probabilities...")
        pi_yw_base, cnt_yw_base = await get_log_probs(session, gateway_url, tokenizer, [s0]*m, questions, yw_list, model_name)
        pi_yl_base, cnt_yl_base = await get_log_probs(session, gateway_url, tokenizer, [s0]*m, questions, yl_list, model_name)
        
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
            
            pi_yw_attr, cnt_yw_attr = await get_log_probs(session, gateway_url, tokenizer, [system]*m, questions, yw_list, model_name)
            pi_yl_attr, cnt_yl_attr = await get_log_probs(session, gateway_url, tokenizer, [system]*m, questions, yl_list, model_name)
            
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

async def evaluate_accuracy(gateway_url: str, test_data: List[Dict[str, str]], p: np.ndarray, tokenizer, model_name: str, base_prompt: str, attribute_prompts: List[str]) -> float:
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
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Get base log probabilities
        print("Computing base log probabilities for test data...")
        chosen_base_probs, chosen_base_counts = await get_log_probs(session, gateway_url, tokenizer, [base_prompt]*n, prompts, chosen, model_name)
        rejected_base_probs, rejected_base_counts = await get_log_probs(session, gateway_url, tokenizer, [base_prompt]*n, prompts, rejected, model_name)
        
        # Initialize drift scores
        drift_scores = torch.zeros(n, dtype=torch.float32)
        
        # Process each attribute
        for i, attr_prompt in enumerate(attribute_prompts):
            if p[i] == 0:
                continue
                
            print(f"Processing test attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
            
            chosen_attr_probs, chosen_attr_counts = await get_log_probs(session, gateway_url, tokenizer, [attr_prompt]*n, prompts, chosen, model_name)
            rejected_attr_probs, rejected_attr_counts = await get_log_probs(session, gateway_url, tokenizer, [attr_prompt]*n, prompts, rejected, model_name)
            
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