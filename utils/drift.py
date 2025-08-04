import torch
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from vllm import LLM, SamplingParams
import gc
import cvxpy as cp

def l1_solve(d_mean, l1_lambda):
    """Solve elastic net optimization problem"""
    p_var = cp.Variable(len(d_mean))
    constraints = [cp.norm2(p_var) <= 1]
    
    objective = cp.Maximize(d_mean @ p_var - l1_lambda * cp.norm1(p_var))
    problem = cp.Problem(objective, constraints)
    
    problem.solve()
    
    if p_var.value is None:
        print(f"Optimization failed for L1={l1_lambda}, using normalization fallback")
        # Use simple normalization as fallback
        current_norm = np.linalg.norm(d_mean, ord=1)
        if current_norm > 1:
            p = d_mean * (1 / current_norm)
        else:
            p = d_mean.copy()
        return p
    else:
        return p_var.value

def approximate(data, pi, tokenizer, s0: str, s_list: list[str], l1_lambda, l2_lambda=1, device=None):
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    questions, yw_list, yl_list = zip(*data)

    for i, system in enumerate(s_list):
        pi_yw_attr, pi_yw_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yw_list, device=device)
        pi_yl_attr, pi_yl_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yl_list, device=device)
        pi_yw_base, pi_yw_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yw_list, device=device)
        pi_yl_base, pi_yl_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yl_list, device=device)

        W[:, i] = torch.tensor(pi_yw_attr, device=device) / torch.tensor(pi_yw_attr_counts, device=device) - torch.tensor(pi_yw_base, device=device) / torch.tensor(pi_yw_base_counts, device=device)
        L[:, i] = torch.tensor(pi_yl_attr, device=device) / torch.tensor(pi_yl_attr_counts, device=device) - torch.tensor(pi_yl_base, device=device) / torch.tensor(pi_yl_base_counts, device=device)

    d = torch.mean(W - L, dim=0).cpu().numpy()
    return l1_solve(d, l1_lambda)

def get_training_matrix(data, pi, tokenizer, s0: str, s_list: list[str], device=None):
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    # Extract data once
    questions, yw_list, yl_list = zip(*data)
    
    # Compute base probabilities ONCE (outside the loop)
    pi_yw_base, pi_yw_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yw_list, device=device)
    pi_yl_base, pi_yl_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yl_list, device=device)
    
    # Convert to tensors once
    yw_base_tensor = torch.tensor(pi_yw_base, device=device, dtype=torch.float32) / torch.tensor(pi_yw_base_counts, device=device, dtype=torch.float32)
    yl_base_tensor = torch.tensor(pi_yl_base, device=device, dtype=torch.float32) / torch.tensor(pi_yl_base_counts, device=device, dtype=torch.float32)

    # Loop through attributes
    for i, system in enumerate(s_list):
        pi_yw_attr, pi_yw_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yw_list, device=device)
        pi_yl_attr, pi_yl_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yl_list, device=device)
        
        # Convert to tensors with consistent dtype
        yw_attr_tensor = torch.tensor(pi_yw_attr, device=device, dtype=torch.float32) / torch.tensor(pi_yw_attr_counts, device=device, dtype=torch.float32)
        yl_attr_tensor = torch.tensor(pi_yl_attr, device=device, dtype=torch.float32) / torch.tensor(pi_yl_attr_counts, device=device, dtype=torch.float32)
        
        W[:, i] = yw_attr_tensor - yw_base_tensor
        L[:, i] = yl_attr_tensor - yl_base_tensor

    return W - L

def get_approximation_accuracy(data, model_ds, p, base_prompt, attribute_prompts, device, tokenizer, batch_size=8):
    """
    Evaluate approximation accuracy using the learned p-vector.
    
    Args:
        data: list of (question, chosen, rejected) tuples
        model_ds: model for scoring
        p: learned drift vector
        base_prompt: base system prompt
        attribute_prompts: list of attribute prompts
        device: torch device
        tokenizer: tokenizer
    """

    questions, yw_list, yl_list = zip(*data)
    n = len(data)

    # Get base log probabilities
    print("Computing base log probabilities...")
    yw_base_probs, yw_base_counts = get_log_probs(model_ds, tokenizer, [base_prompt] * n, questions, yw_list, device)
    yl_base_probs, yl_base_counts = get_log_probs(model_ds, tokenizer, [base_prompt] * n, questions, yl_list, device)

    # Initialize drift scores for each example
    drift_scores = torch.zeros(n, device=device)

    # Process each attribute prompt individually
    for i, attribute_prompt in enumerate(attribute_prompts):
        if p[i] == 0:
            print(f"Skipping attribute {i} (p={p[i]})")
            continue
            
        print(f"Processing attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
        
        # Get log probabilities for this attribute prompt
        yw_attr_probs, yw_attr_counts = get_log_probs(model_ds, tokenizer, [attribute_prompt] * n, questions, yw_list, device)
        yl_attr_probs, yl_attr_counts = get_log_probs(model_ds, tokenizer, [attribute_prompt] * n, questions, yl_list, device)
        
        # Convert to tensors
        yw_attr_tensor = torch.tensor(yw_attr_probs, device=device) / torch.tensor(yw_attr_counts, device=device)
        yl_attr_tensor = torch.tensor(yl_attr_probs, device=device) / torch.tensor(yl_attr_counts, device=device)
        yw_base_tensor = torch.tensor(yw_base_probs, device=device) / torch.tensor(yw_base_counts, device=device)
        yl_base_tensor = torch.tensor(yl_base_probs, device=device) / torch.tensor(yl_base_counts, device=device)
        
        # Compute drift contribution for this attribute
        # drift = p[i] * ((yw_attr - yw_base) - (yl_attr - yl_base))
        attribute_drift = p[i] * ((yw_attr_tensor - yw_base_tensor) - (yl_attr_tensor - yl_base_tensor))
        
        # Add to total drift scores
        drift_scores += attribute_drift

    # Count how many examples have positive drift scores (chosen > rejected)
    correct = (drift_scores > 0).sum().item()
    accuracy = correct / n
    
    print(f"Accuracy: {accuracy:.4f} ({correct}/{n})")
    return accuracy

def get_scores(data, model, p, base_prompt, attribute_prompts, device, tokenizer):
    """
    Compute drift scores for multiple outputs per prompt.
    
    Args:
        data: List of (prompt, output_list) tuples
              where output_list contains n outputs for each prompt
        model: vLLM model
        p: drift vector weights
        base_prompt: base system prompt
        attribute_prompts: list of attribute system prompts
        device: torch device
        tokenizer: tokenizer
    
    Returns:
        torch.Tensor: m x n matrix where m = number of prompts, n = number of outputs per prompt
    """
    m = len(data)  # number of prompts
    n = len(data[0][1])  # number of outputs per prompt (assuming all have same length)
    
    # Flatten all data for batch processing
    flat_questions = []
    flat_outputs = []
    
    for prompt, output_list in data:
        for output in output_list:
            flat_questions.append(prompt)
            flat_outputs.append(output)
    
    total_items = len(flat_outputs)  # m * n
    print(f"Processing {m} prompts with {n} outputs each ({total_items} total items)")
    
    # Get base log probabilities for all flattened items
    print("Computing base log probabilities...")
    base_probs, base_counts = get_log_probs(
        model, tokenizer, [base_prompt] * total_items, 
        flat_questions, flat_outputs, device
    )
    base_tensor = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Initialize drift scores for all items
    drift_scores = torch.zeros(total_items, device=device)
    
    # Process each attribute prompt individually
    for i, attribute_prompt in enumerate(attribute_prompts):
        if p[i] == 0:
            print(f"Skipping attribute {i} (p={p[i]})")
            continue
            
        print(f"Processing attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
        
        # Get log probabilities for this attribute prompt
        attr_probs, attr_counts = get_log_probs(
            model, tokenizer, [attribute_prompt] * total_items, 
            flat_questions, flat_outputs, device
        )
        
        # Convert to tensors
        attr_tensor = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Compute drift contribution for this attribute
        attribute_drift = p[i] * (attr_tensor - base_tensor)
        
        # Add to total drift scores
        drift_scores += attribute_drift
    
    # Reshape back to m x n matrix
    score_matrix = drift_scores.view(m, n)
    
    return score_matrix

def get_log_probs(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    input_ids = []
    ns = []
    completion_ids = []
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        # Apply chat template to get prompt tokens
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        ns.append(len(prompt_ids))
        # Tokenize completion without skipping tokens
        completion_ids_i = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        input_ids_i = prompt_ids + completion_ids_i + [tokenizer.eos_token_id]
        input_ids.append(input_ids_i)
        completion_ids.append(completion_ids_i)
    sampling_params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=temperature,
    )

    outputs = model.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    log_probs = []
    for compl, out, n in zip(input_ids, outputs, ns):
        logprobs = [
            (lxi[xi].logprob)
            for xi, lxi in zip(
                compl[1:],
                out.prompt_logprobs[1:],
            )
        ][n:]
        log_probs.append(sum(logprobs))

    token_counts = [len(compl) for compl in completion_ids]
    return log_probs, token_counts

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
        # Simplified version without caching - more reliable
        prompt_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(input_ids.device)
        
        # Concatenate system prompt with current token
        input_step = torch.cat([prompt_ids, input_ids[:, -1:]], dim=1)
        attention_mask = torch.ones_like(input_step)

        with torch.no_grad():
            out = self.small_model(
                input_step,
                attention_mask=attention_mask,
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