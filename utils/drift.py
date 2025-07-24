import torch
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from typing import Optional
from tqdm import tqdm
import cvxpy as cp
from vllm import LLM, SamplingParams
import gc

def log_prob(pi, ys, systems, questions, device, tokenizer, batch_size=8, show_progress=True):
    log_probs = []
    token_counts = []
    pi.eval()

    loop = range(0, len(ys), batch_size)
    if show_progress:
        from tqdm import tqdm
        loop = tqdm(loop, desc="log_prob batches")

    with torch.no_grad():
        for i in loop:
            batch_ys = ys[i:i + batch_size]
            batch_systems = systems[i:i + batch_size]
            batch_questions = questions[i:i + batch_size]

            full_texts = []
            prompt_texts = []

            for y, sys_prompt, question in zip(batch_ys, batch_systems, batch_questions):
                full_messages = [
                    {"role": "system", "content": sys_prompt.strip()},
                    {"role": "user", "content": question.strip()},
                    {"role": "assistant", "content": y.lstrip()}
                ]

                prompt_messages = [
                    {"role": "system", "content": sys_prompt.strip()},
                    {"role": "user", "content": question.strip()}
                ]

                full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
                prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

                full_texts.append(full_text)
                prompt_texts.append(prompt_text)

            encoded_input = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            encoded_prompt = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)

            input_ids = encoded_input.input_ids
            prompt_ids = encoded_prompt.input_ids
            attention_mask = encoded_input.attention_mask

            outputs = pi(input_ids, attention_mask=attention_mask).logits
            probs = torch.nn.functional.log_softmax(outputs, dim=-1)

            max_len = input_ids.shape[1]
            prompt_ids_padded = torch.full_like(input_ids, tokenizer.pad_token_id)
            prompt_ids_padded[:, :prompt_ids.shape[1]] = prompt_ids

            diffs = (input_ids != prompt_ids_padded)
            split_indices = diffs.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
            all_equal = (~diffs).all(dim=1)
            split_indices[all_equal] = prompt_ids.shape[1]

            for j in range(input_ids.shape[0]):
                start = split_indices[j].item()
                target_ids = input_ids[j, start:]
                if start > 0:
                    token_probs = probs[j, start-1:-1].gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                else:
                    token_probs = probs[j, :len(target_ids)].gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                log_probs.append(token_probs.sum().item())
                token_counts.append(len(target_ids))

    return log_probs, token_counts

def approximate(data, pi, tokenizer, s0: str, s_list: list[str], device, batch_size=8):
    pi.eval()
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    for i, system in enumerate(s_list):
        print(i)
        questions = [q for q, _, _ in data]
        yw_list = [yw for _, yw, _ in data]
        yl_list = [yl for _, _, yl in data]

        pi_yw_attr, _ = log_prob(pi, yw_list, [system]*m, questions, device, tokenizer, batch_size)
        pi_yl_attr, _ = log_prob(pi, yl_list, [system]*m, questions, device, tokenizer, batch_size)
        pi_yw_base, _ = log_prob(pi, yw_list, [s0]*m, questions, device, tokenizer, batch_size)
        pi_yl_base, _ = log_prob(pi, yl_list, [s0]*m, questions, device, tokenizer, batch_size)

        W[:, i] = torch.tensor(pi_yw_attr, device=device) - torch.tensor(pi_yw_base, device=device)
        L[:, i] = torch.tensor(pi_yl_attr, device=device) - torch.tensor(pi_yl_base, device=device)

    d = torch.sum(W - L, dim=0)
    p = d / torch.norm(d, p=2)
    return p

def find_sublist(sub, full):
    """Find the start index of sub in full, or return -1 if not found."""
    for i in range(len(full) - len(sub) + 1):
        if full[i:i+len(sub)] == sub:
            return i
    return -1

def get_log_probs_vllm(systems, questions, completions, llm, tokenizer, batch_size=8):
    assert len(systems) == len(questions) == len(completions)

    log_probs = []
    token_counts = []
    n = len(systems)
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_systems = systems[batch_start:batch_end]
        batch_questions = questions[batch_start:batch_end]
        batch_completions = completions[batch_start:batch_end]

        batch_full_texts = []
        batch_prompt_texts = []
        for sys, q, c in zip(batch_systems, batch_questions, batch_completions):
            full_messages = [
                {"role": "system", "content": sys.strip()},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": c.lstrip()}
            ]
            prompt_messages = [
                {"role": "system", "content": sys.strip()},
                {"role": "user", "content": q.strip()}
            ]
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            batch_full_texts.append(full_text)
            batch_prompt_texts.append(prompt_text)

        batch_prompt_ids_list = tokenizer(batch_prompt_texts, return_tensors="pt", padding=True).input_ids.tolist()
        batch_full_ids_list = tokenizer(batch_full_texts, return_tensors="pt", padding=True).input_ids.tolist()

        sampling_params = SamplingParams(
            temperature=0.0,
            prompt_logprobs=1,
            max_tokens=1
        )
        batch_outputs = llm.generate(batch_full_texts, sampling_params)

        for idx, (out, prompt_ids, full_ids) in enumerate(zip(batch_outputs, batch_prompt_ids_list, batch_full_ids_list)):
            
            prompt_len = len(out.prompt_token_ids)
            if full_ids[:prompt_len] != out.prompt_token_ids:
                print(full_ids[:prompt_len])
                print(out.prompt_token_ids)
                assert False
            log_prob_dict = out.prompt_logprobs
            # Robustly find where the prompt ends in the full input
            start = find_sublist(prompt_ids, full_ids)
            if start == -1:
                print(f"Warning: Prompt tokens not found as prefix in full tokens for example {batch_start + idx}. Using fallback split.")
                start = prompt_ids.index(tokenizer.eos_token_id)
            else:
                start += len(prompt_ids)
            target_ids = full_ids[start:]
            completion_log_probs = []
            tokens_found = 0
            for i, token_id in enumerate(target_ids, start=start):
                if i >= len(log_prob_dict):
                    break
                logprob_entry = log_prob_dict[i].get(token_id)
                if logprob_entry is not None:
                    completion_log_probs.append(logprob_entry.logprob)
                    tokens_found += 1
                else:
                    print(f"Token ID {token_id} not found in prompt_logprobs at position {i} (example {batch_start + idx})")
                    assert False
            if len(completion_log_probs) == 0:
                print(f"No log probs found for example {batch_start + idx}")
                print(f"Full ids: {full_ids}")
                print(f"Prompt ids: {prompt_ids}")
                print(f"Target ids: {target_ids}")
                print(f"Full text: {out.prompt_token_ids}")
                assert False
            
            log_probs.append(sum(completion_log_probs))
            token_counts.append(tokens_found)
        # Explicitly delete and collect garbage
        del batch_outputs
        gc.collect()
    return log_probs, token_counts

def approximate_l1(data, llm, tokenizer, s0: str, s_list: list[str], device, lambda_reg=0.1, batch_size=8):
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    questions, yw_list, yl_list = zip(*data)

    # === Step 1: Compute base log probs (prompt = s0) ===
    pi_yw_base, pi_yw_base_counts = get_log_probs_vllm([s0] * m, list(questions), list(yw_list), llm, tokenizer, batch_size=batch_size)
    pi_yl_base, pi_yl_base_counts = get_log_probs_vllm([s0] * m, list(questions), list(yl_list), llm, tokenizer, batch_size=batch_size)

    # === Step 2: Construct one big batch for all attribute prompts ===
    all_systems_yw = []
    all_questions_yw = []
    all_completions_yw = []

    all_systems_yl = []
    all_questions_yl = []
    all_completions_yl = []

    for system in s_list:
        all_systems_yw.extend([system] * m)
        all_questions_yw.extend(questions)
        all_completions_yw.extend(yw_list)

        all_systems_yl.extend([system] * m)
        all_questions_yl.extend(questions)
        all_completions_yl.extend(yl_list)

    # === Step 3: Batch inference ===
    pi_yw_attr_all, pi_yw_attr_all_counts = get_log_probs_vllm(all_systems_yw, all_questions_yw, all_completions_yw, llm, tokenizer, batch_size=batch_size)
    pi_yl_attr_all, pi_yl_attr_all_counts = get_log_probs_vllm(all_systems_yl, all_questions_yl, all_completions_yl, llm, tokenizer, batch_size=batch_size)

    # === Step 4: Reconstruct W and L ===
    for i in range(k):
        start = i * m
        end = (i + 1) * m
        W[:, i] = torch.tensor(pi_yw_attr_all[start:end], device=device) - torch.tensor(pi_yw_base, device=device)
        L[:, i] = torch.tensor(pi_yl_attr_all[start:end], device=device) - torch.tensor(pi_yl_base, device=device)

    # === Step 5: Solve the optimization ===
    a = torch.sum(W - L, dim=0).cpu().numpy()

    p_var = cp.Variable(len(a))
    constraints = [cp.norm2(p_var) <= 1.0]
    objective = cp.Maximize(p_var @ a - lambda_reg * cp.norm1(p_var))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    p = torch.tensor(p_var.value, device=device, dtype=torch.float32)
    if p.norm() > 0:
        p = p / p.norm(p=2)

    return p


class DriftLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        b: float,
        small_model,
        tokenizer,
        base_prompt: str,
        attribute_prompts: list[str],
        weights: list[float],
        use_cache: bool = True,
    ):
        self.b = b
        self.small_model = small_model
        self.tokenizer = tokenizer
        self.base_prompt = base_prompt
        self.attribute_prompts = attribute_prompts
        self.weights = weights
        self.use_cache = use_cache

        self.refs = {prompt: {
            "input_ids": None,
            "attention_mask": None,
            "past_key_values": None,
            "first_pass": True
        } for prompt in [base_prompt] + attribute_prompts}

    def get_small_logits(self, input_ids, prompt):
        ref = self.refs[prompt]

        if ref["first_pass"]:
            if ref["input_ids"] is None:
                prompt_text = self.tokenizer.apply_chat_template([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "__Q__"},
                    {"role": "assistant", "content": ""}
                ], tokenize=False, add_generation_prompt=True)

                prompt_text = prompt_text.replace("__Q__", "")
                prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(input_ids.device)

                ref["input_ids"] = prompt_ids
                ref["attention_mask"] = torch.ones_like(prompt_ids)
                ref["past_key_values"] = None
                ref["first_pass"] = False

        prompt_ids = ref["input_ids"]
        attention_mask = ref["attention_mask"]

        if self.use_cache:
            input_step = input_ids[:, -1:]
        else:
            input_step = torch.cat([prompt_ids, input_ids[:, -1:]], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids[:, -1:])], dim=1)

        out = self.small_model(
            input_step,
            attention_mask=attention_mask,
            use_cache=self.use_cache,
            past_key_values=ref["past_key_values"]
        )

        ref["past_key_values"] = out.get("past_key_values", None)
        ref["attention_mask"] = attention_mask

        return out.logits[:, -1]

    def __call__(self, input_ids, aligned_logits):
        h0_small = self.get_small_logits(input_ids, self.base_prompt)

        drift = torch.zeros_like(h0_small)
        for w, attr_prompt in zip(self.weights, self.attribute_prompts):
            if w == 0:
                continue
            hi_small = self.get_small_logits(input_ids, attr_prompt)
            drift += w * (hi_small - h0_small)

        return aligned_logits + drift / self.b

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
    yw_base_probs, _ = log_prob(model_ds, yw_list, [base_prompt] * n, questions, device, tokenizer, batch_size=batch_size)
    yl_base_probs, _ = log_prob(model_ds, yl_list, [base_prompt] * n, questions, device, tokenizer, batch_size=batch_size)

    # Initialize drift scores for each example
    drift_scores = torch.zeros(n, device=device)

    # Process each attribute prompt individually
    for i, attribute_prompt in enumerate(attribute_prompts):
        if p[i] == 0:
            print(f"Skipping attribute {i} (p={p[i]})")
            continue
            
        print(f"Processing attribute {i+1}/{len(attribute_prompts)}: p={p[i]:.4f}")
        
        # Get log probabilities for this attribute prompt
        yw_attr_probs, _ = log_prob(model_ds, yw_list, [attribute_prompt] * n, questions, device, tokenizer, batch_size=batch_size, show_progress=False)
        yl_attr_probs, _ = log_prob(model_ds, yl_list, [attribute_prompt] * n, questions, device, tokenizer, batch_size=batch_size, show_progress=False)
        
        # Convert to tensors
        yw_attr_tensor = torch.tensor(yw_attr_probs, device=device)
        yl_attr_tensor = torch.tensor(yl_attr_probs, device=device)
        yw_base_tensor = torch.tensor(yw_base_probs, device=device)
        yl_base_tensor = torch.tensor(yl_base_probs, device=device)
        
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

# def get_approximation_accuracy(data, model_ds, p, base_prompt, attribute_prompts, device, tokenizer, batch_size=8):
#     """
#     Evaluate approximation accuracy using the learned p-vector.
    
#     Args:
#         data: list of (question, chosen, rejected) tuples
#         model_ds: model for scoring
#         p: learned drift vector
#         base_prompt: base system prompt
#         attribute_prompts: list of attribute prompts
#         device: torch device
#         tokenizer: tokenizer
#     """

#     questions, yw_list, yl_list = zip(*data)
#     k, n = len(attribute_prompts), len(data)

#     yw_base_probs, yw_base_counts = log_prob(model_ds, yw_list, [base_prompt] * n, questions, device, tokenizer, batch_size=batch_size)
#     yl_base_probs, yl_base_counts = log_prob(model_ds, yl_list, [base_prompt] * n, questions, device, tokenizer, batch_size=batch_size)

#     all_systems_yw = []
#     all_questions_yw = []
#     all_completions_yw = []

#     all_systems_yl = []
#     all_questions_yl = []
#     all_completions_yl = []

#     p_sparse = []

#     for pi, system in zip(p, attribute_prompts):
#         if pi == 0:
#             continue
#         p_sparse.append(pi)
#         all_systems_yw.extend([system] * n)
#         all_questions_yw.extend(questions)
#         all_completions_yw.extend(yw_list)

#         all_systems_yl.extend([system] * n)
#         all_questions_yl.extend(questions)
#         all_completions_yl.extend(yl_list)

#     p_sparse = torch.tensor(p_sparse, device=device, dtype=torch.float32)
    
#     pi_yw_attr_all, yw_attr_counts = log_prob(model_ds, all_completions_yw, all_systems_yw, all_questions_yw, device, tokenizer, batch_size=batch_size)
#     pi_yl_attr_all, yl_attr_counts = log_prob(model_ds, all_completions_yl, all_systems_yl, all_questions_yl, device, tokenizer, batch_size=batch_size)

#     yw_attr_probs = torch.tensor(pi_yw_attr_all, device=device, dtype=torch.float32)
#     yl_attr_probs = torch.tensor(pi_yl_attr_all, device=device, dtype=torch.float32)

#     yw_base_probs = torch.tensor(yw_base_probs, device=device, dtype=torch.float32).view(n, 1)
#     yl_base_probs = torch.tensor(yl_base_probs, device=device, dtype=torch.float32).view(n, 1)

#     # Reshape to (len(p_sparse), n) then transpose to (n, len(p_sparse))
#     yw_attr_probs = yw_attr_probs.view(n, len(p_sparse))
#     yl_attr_probs = yl_attr_probs.view(n, len(p_sparse))

#     attr_scores = (yw_attr_probs - yw_base_probs - yl_attr_probs + yl_base_probs) @ p_sparse.view(len(p_sparse), 1)
#     attr_scores = (attr_scores > 0).sum() / n
#     return attr_scores.item()

def best_of_n_decode(n, question, model_bs, model_ds, base_prompt, attribute_prompts, device, tokenizer, logits_processor, b=1.0):
    """
    n: number of generations per prompt
    question: question to generate for
    model_bs: the model to generate with
    model_ds: the model to score with
    base_prompt: base system prompt
    attribute_prompts: list of attribute prompts
    device: torch device
    tokenizer: tokenizer for the model
    logits_processor: function/class for drift
    b: drift parameter
    """
    model_ds.eval()
    model_bs.eval()

    prompt_text = tokenizer.apply_chat_template([
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": question}
    ], tokenize=False, add_generation_prompt=True)

    # Generate n completions
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
    generations = []
    lengths = []
    for i in range(n):
        output = model_bs.generate(
            **inputs, 
            logits_processor=LogitsProcessorList([logits_processor]),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            )
        generated = output[0][inputs['input_ids'].shape[1]:]
        lengths.append(generated.shape[0])
        decoded_output = tokenizer.decode(generated, skip_special_tokens=True)
        generations.append(decoded_output)
        print(decoded_output)
    
    # Find the best completion
    best_prompt = None
    best_score = -float("inf")
    for i in range(n):
        curr_log_prob, curr_length = log_prob(model_ds, [generations[i]], [base_prompt], [question], device, tokenizer)
        curr_score = curr_log_prob[0] / curr_length[0]
        if curr_score > best_score:
            best_score = curr_score
            best_prompt = generations[i]
    
    return best_prompt

def drift_score_bon(data, pi, tokenizer, s0: str, s_list: list[str], p, device, batch_size=8):

    pi.eval()
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=device)
    all_scores = []
    
    for question, outputs in tqdm(data, desc="Computing drift scores"):
        m = len(outputs)
        k = len(s_list)
        W = torch.zeros(m, k, device=device)
        for i, system in enumerate(s_list):
            if p[i] == 0:
                continue
            questions = [question] * m
            pi_y_attr, token_counts_attr = log_prob(pi, outputs, [system]*m, questions, device, tokenizer, batch_size)
            pi_y_base, token_counts_base = log_prob(pi, outputs, [s0]*m, questions, device, tokenizer, batch_size)
            
            pi_y_attr_norm = [log_prob / token_count for log_prob, token_count in zip(pi_y_attr, token_counts_attr)]
            pi_y_base_norm = [log_prob / token_count for log_prob, token_count in zip(pi_y_base, token_counts_base)]
            
            W[:, i] = p[i] * (torch.tensor(pi_y_attr_norm, device=device) - torch.tensor(pi_y_base_norm, device=device))
        scores = W.sum(dim=1)
        all_scores.append(scores.tolist())
    return all_scores

def drift_score_bon_batched(data, pi, tokenizer, s0: str, s_list: list[str], p, device, batch_size=8, batch_size_outer=8):
    """
    Batched version of drift_score_bon. Processes multiple (question, outputs) pairs at once for efficiency.
    Args:
        data: list of (question, outputs) pairs
        pi: model
        tokenizer: tokenizer
        s0: base system prompt
        s_list: list of attribute prompts
        p: drift vector
        device: torch device
        batch_size: batch size for log_prob
        batch_size_outer: number of (question, outputs) pairs to process at once
    Returns:
        all_scores: list of scores for each (question, outputs) pair
    """
    from tqdm import tqdm
    pi.eval()
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=device)
    all_scores = []
    for batch_start in tqdm(range(0, len(data), batch_size_outer), desc="Computing drift scores (batched)"):
        batch = data[batch_start:batch_start+batch_size_outer]
        # Flatten outputs and keep track of indices
        flat_outputs = []
        flat_questions = []
        output_lens = []
        for question, outputs in batch:
            flat_outputs.extend(outputs)
            flat_questions.extend([question] * len(outputs))
            output_lens.append(len(outputs))
        if not flat_outputs:
            continue
        W = torch.zeros(len(flat_outputs), len(s_list), device=device)
        for i, system in enumerate(s_list):
            if p[i] == 0:
                print(f"Skipping {system} because p[{i}] = 0")
                continue
            pi_y_attr, token_counts_attr = log_prob(pi, flat_outputs, [system]*len(flat_outputs), flat_questions, device, tokenizer, batch_size)
            pi_y_base, token_counts_base = log_prob(pi, flat_outputs, [s0]*len(flat_outputs), flat_questions, device, tokenizer, batch_size)
            pi_y_attr_norm = [log_prob / token_count for log_prob, token_count in zip(pi_y_attr, token_counts_attr)]
            pi_y_base_norm = [log_prob / token_count for log_prob, token_count in zip(pi_y_base, token_counts_base)]
            W[:, i] = p[i] * (torch.tensor(pi_y_attr_norm, device=device) - torch.tensor(pi_y_base_norm, device=device))
        # Unflatten scores
        idx = 0
        for l in output_lens:
            scores = W[idx:idx+l, :].sum(dim=1)
            all_scores.append(scores.tolist())
            idx += l
    return all_scores