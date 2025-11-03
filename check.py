import json
import numpy as np
from transformers import AutoTokenizer
import aiohttp
import torch
from typing import List, Dict, Tuple
import asyncio
import os

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "5"))
REQUEST_BATCH_SIZE = int(os.getenv("REQUEST_BATCH_SIZE", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

with open("data/PERSONA_testing/user1_test.json", "r") as f:
    test_data = json.load(f)


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


async def main():
    attribute_prompts = [
        "A renowned futurist and technological trendsetter who provides insights on the impact of predictive models on various industries",
        "Peter Keane's long-time frenemy and a die-hard Gaelic football fan",
        "A talented teenager with a passion for cars and a desire to become a professional drifter",
        "A mayor of a town near the military base, who appreciates the officer's efforts to keep the community informed",
        "A nuclear physicist who lived through and contributed to the nuclear advancements in the Cold War era",
        "An art historian who enjoys sharing thoughts on cultural topics and seeks legal opinions on art restitution cases",
        "A programming instructor who is knowledgeable in Git and emphasizes clear instruction.",
        "a veteran who experienced training with Pepper spray",
    ]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    base_prompt = "You are a helpful assistant."

    print(await evaluate_accuracy("http://localhost:8080", test_data, np.array([0.008719060570001602, 0.02320767752826214, 13.614852905273438, 0.0009940010495483875, -0.0015057716518640518, 0.034328166395425797, 0.004022639244794846, 0.015323520638048649]), tokenizer, "meta-llama/Llama-3.2-1B-Instruct", base_prompt, attribute_prompts))

if __name__ == "__main__":
    asyncio.run(main())