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
    """
    Closed-form solution to: maximize d^T p - lambda * ||p||_1  s.t. ||p||_2 <= 1
    """
    d = np.asarray(d_mean, dtype=float)
    # soft-threshold
    z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
    norm = np.linalg.norm(z, ord=2)
    if norm == 0.0:
        return np.zeros_like(d)
    return z / norm

def approximate(data, pi, tokenizer, s0: str, s_list: list[str], l1_lambda, l2_lambda=1, device=None):
    # data: list of (question, y_w, y_l)
    m, k = len(data), len(s_list)
    questions, yw_list, yl_list = zip(*data)

    # Compute base once
    pi_yw_base, cnt_yw_base = get_log_probs(pi, tokenizer, [s0]*m, questions, yw_list, device=device)
    pi_yl_base, cnt_yl_base = get_log_probs(pi, tokenizer, [s0]*m, questions, yl_list, device=device)

    pi_yw_base = torch.tensor(pi_yw_base, device=device, dtype=torch.float32)
    cnt_yw_base = torch.tensor(cnt_yw_base, device=device, dtype=torch.float32)
    pi_yl_base = torch.tensor(pi_yl_base, device=device, dtype=torch.float32)
    cnt_yl_base = torch.tensor(cnt_yl_base, device=device, dtype=torch.float32)

    # safe average log-probs
    eps = 1e-12
    yw_base_avg = pi_yw_base / torch.clamp(cnt_yw_base, min=eps)
    yl_base_avg = pi_yl_base / torch.clamp(cnt_yl_base, min=eps)

    # Build X = (W - L) with columns per attribute
    X = torch.zeros((m, k), device=device, dtype=torch.float32)

    for j, system in enumerate(s_list):
        pi_yw_attr, cnt_yw_attr = get_log_probs(pi, tokenizer, [system]*m, questions, yw_list, device=device)
        pi_yl_attr, cnt_yl_attr = get_log_probs(pi, tokenizer, [system]*m, questions, yl_list, device=device)

        pi_yw_attr = torch.tensor(pi_yw_attr, device=device, dtype=torch.float32)
        cnt_yw_attr = torch.tensor(cnt_yw_attr, device=device, dtype=torch.float32)
        pi_yl_attr = torch.tensor(pi_yl_attr, device=device, dtype=torch.float32)
        cnt_yl_attr = torch.tensor(cnt_yl_attr, device=device, dtype=torch.float32)

        yw_attr_avg = pi_yw_attr / torch.clamp(cnt_yw_attr, min=eps)
        yl_attr_avg = pi_yl_attr / torch.clamp(cnt_yl_attr, min=eps)

        # column j: (yw_attr - yw_base) - (yl_attr - yl_base)
        X[:, j] = (yw_attr_avg - yw_base_avg) - (yl_attr_avg - yl_base_avg)

    # Option A (your original): mean over samples
    d = X.mean(dim=0).detach().cpu().numpy()

    # Optional: standardize columns before mean to avoid variance dominance
    # col_std = X.std(dim=0).clamp_min(1e-8)
    # d = (X / col_std).mean(dim=0).detach().cpu().numpy()

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