import torch
import numpy as np
import random
from vllm import LLM, SamplingParams
from typing import List, Tuple, Optional
import math
from transformers import AutoTokenizer

def qalign(
    model: LLM,
    tokenizer,
    question: str,
    reward_fn,
    base_prompt: str = "You are an AI assistant.",
    num_steps: int = 100,
    beta: float = 1.0,
    initial_response: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.8,
    device: str = "cuda"
) -> List[str]:
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
        List of accepted samples from the MCMC chain
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
    
    return accepted_samples


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