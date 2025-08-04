import torch
import numpy as np
import random
from vllm import LLM, SamplingParams
from typing import List, Tuple, Optional
import math
from transformers import AutoTokenizer
from attribute_prompts import base_prompt

def rouge_l(candidate, reference):
    """
    Compute ROUGE-L score using Longest Common Subsequence (LCS).
    Returns dict with precision, recall, and f1 scores.
    """
    def lcs_length(x, y):
        """Compute the length of the longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Tokenize by splitting on whitespace
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    if len(candidate_tokens) == 0 and len(reference_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs_len = lcs_length(candidate_tokens, reference_tokens)
    
    precision = lcs_len / len(candidate_tokens) if len(candidate_tokens) > 0 else 0.0
    recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}

def qalign_mbr_rouge_l(candidates):
    """
    Select the candidate that maximizes average ROUGE-L F1 score against all candidates.
    O(T^2) algorithm where T is the number of candidates.
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    best_candidate = None
    best_score = -1
    
    for candidate in candidates:
        total_utility = 0
        for other_candidate in candidates:
            utility = rouge_l(candidate, other_candidate)["f1"]
            total_utility += utility
        
        avg_utility = total_utility / len(candidates)
        if avg_utility > best_score:
            best_score = avg_utility
            best_candidate = candidate
    
    return best_candidate

def qalign(
    model: LLM,
    tokenizer,
    question: str,
    reward_fn,
    base_prompt: str = base_prompt,
    num_steps: int = 100,
    beta: float = 1.0,
    initial_response: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.8,
    device: str = "cuda"
) -> str:
    """
    Args:
        model: vLLM model for generation
        tokenizer: tokenizer for the model
        question: input question/prompt
        reward_fn: function that takes (question, response) and returns reward score
        base_prompt: base system prompt
        num_steps: number of MCMC steps T
        beta: temperature parameter for acceptance probability
        initial_response: initial response (if None, generates one)
        max_tokens: maximum tokens to generate
        temperature: sampling temperature for generation
        device: device to use
        
    Returns:
        Single response selected using MBR ROUGE-L from accepted samples
    """
    
    # Step 1: Initialize the chain with y^0
    if initial_response is None:
        print("Generating initial response...")
        initial_response = generate_response(model, tokenizer, base_prompt, question, max_tokens, temperature)
    
    current_response = initial_response
    current_reward = reward_fn(question, current_response)
    
    accepted_samples = [current_response]
    acceptance_count = 0
    
    print(f"Starting MCMC with initial reward: {current_reward:.4f}")
    
    for step in range(num_steps):
        # Step 2: Generate proposal y from q(y | y^t, x)
        # Using QUEST proposal: sample index i uniformly, then complete from that point
        proposal_response = generate_proposal(
            model, tokenizer, base_prompt, question, current_response, max_tokens, temperature
        )
        
        if proposal_response is None:
            continue
            
        # Step 3: Compute acceptance probability
        proposal_reward = reward_fn(question, proposal_response)
        
        # Compute acceptance probability using Eq. 9
        reward_diff = proposal_reward - current_reward
        length_ratio = len(tokenizer.encode(current_response)) / len(tokenizer.encode(proposal_response))
        
        # α_β(y, y^t) = min{1, exp((r(x,y) - r(x,y^t))/β) * |y^t|/|y|}
        log_acceptance_prob = min(0, reward_diff / beta + math.log(length_ratio))
        acceptance_prob = math.exp(log_acceptance_prob)
        
        # Sample random boolean based on acceptance probability
        if random.random() < acceptance_prob:
            # Accept proposal
            current_response = proposal_response
            current_reward = proposal_reward
            accepted_samples.append(current_response)
            acceptance_count += 1
            status = "ACCEPTED"
        else:
            # Reject proposal, stay at current state
            accepted_samples.append(current_response)
            status = "REJECTED"
        
        if (step + 1) % 10 == 0:
            acceptance_rate = acceptance_count / (step + 1)
            print(f"Step {step + 1}/{num_steps}: {status}, "
                  f"Current reward: {current_reward:.4f}, "
                  f"Acceptance rate: {acceptance_rate:.3f}")
    
    final_acceptance_rate = acceptance_count / num_steps
    print(f"MCMC completed. Final acceptance rate: {final_acceptance_rate:.3f}")
    print(f"Selecting best response from {len(accepted_samples)} candidates using MBR ROUGE-L...")
    
    # Use MBR ROUGE-L to select the best response from accepted samples
    best_response = qalign_mbr_rouge_l(accepted_samples)
    
    return best_response


def generate_response(model: LLM, tokenizer, system_prompt: str, question: str, 
                     max_tokens: int, temperature: float) -> str:
    """Generate a complete response to the question."""
    prompt_text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ], tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )
    
    outputs = model.generate([prompt_text], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    
    return response


def generate_proposal(model: LLM, tokenizer, system_prompt: str, question: str, 
                     current_response: str, max_tokens: int, temperature: float) -> Optional[str]:
    """
    Generate a proposal using QUEST method:
    1. Sample index i uniformly from current response
    2. Generate completion from that point
    """
    # Tokenize current response to get its length
    current_tokens = tokenizer.encode(current_response, add_special_tokens=False)
    
    if len(current_tokens) == 0:
        return generate_response(model, tokenizer, system_prompt, question, max_tokens, temperature)
    
    # Step 1: Sample index i uniformly
    i = random.randint(0, len(current_tokens) - 1)
    
    # Step 2: Create prefix up to index i
    prefix_tokens = current_tokens[:i]
    prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
    
    # Step 3: Generate completion from this prefix
    prompt_text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ], tokenize=False, add_generation_prompt=True)
    
    # Add the prefix to the prompt
    full_prompt = prompt_text + prefix_text
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens - i,  # Adjust max tokens based on prefix length
        temperature=temperature,
        top_p=0.9
    )
    
    try:
        outputs = model.generate([full_prompt], sampling_params)
        completion = outputs[0].outputs[0].text.strip()
        
        # Combine prefix with new completion
        proposal = prefix_text + completion
        
        return proposal
    except Exception as e:
        print(f"Error generating proposal: {e}")
        return None