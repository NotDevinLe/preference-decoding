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

def approximate(data, pi, tokenizer, s0: str, s_list: list[str], device):
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    for i, system in enumerate(s_list):
        print(i)
        questions = [q for q, _, _ in data]
        yw_list = [yw for _, yw, _ in data]
        yl_list = [yl for _, _, yl in data]

        pi_yw_attr, pi_yw_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yw_list, device=device)
        pi_yl_attr, pi_yl_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yl_list, device=device)
        pi_yw_base, pi_yw_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yw_list, device=device)
        pi_yl_base, pi_yl_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yl_list, device=device)

        W[:, i] = torch.tensor(pi_yw_attr, device=device) / torch.tensor(pi_yw_attr_counts, device=device) - torch.tensor(pi_yw_base, device=device) / torch.tensor(pi_yw_base_counts, device=device)
        L[:, i] = torch.tensor(pi_yl_attr, device=device) / torch.tensor(pi_yl_attr_counts, device=device) - torch.tensor(pi_yl_base, device=device) / torch.tensor(pi_yl_base_counts, device=device)

    with open("d.pkl", "wb") as f:
        pickle.dump(W-L, f)
    d = torch.mean(W - L, dim=0)
    current_norm = torch.norm(d, p=1)
    if current_norm > 1:
        p = d * (1 / current_norm)
    else:
        p = d
    return p

def get_training_matrix(data, pi, tokenizer, s0: str, s_list: list[str], device):
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    for i, system in enumerate(s_list):
        print(i)
        questions = [q for q, _, _ in data]
        yw_list = [yw for _, yw, _ in data]
        yl_list = [yl for _, _, yl in data]

        pi_yw_attr, pi_yw_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yw_list, device=device)
        pi_yl_attr, pi_yl_attr_counts = get_log_probs(pi, tokenizer, [system]*m, questions, yl_list, device=device)
        pi_yw_base, pi_yw_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yw_list, device=device)
        pi_yl_base, pi_yl_base_counts = get_log_probs(pi, tokenizer, [s0]*m, questions, yl_list, device=device)

        W[:, i] = torch.tensor(pi_yw_attr, device=device) / torch.tensor(pi_yw_attr_counts, device=device) - torch.tensor(pi_yw_base, device=device) / torch.tensor(pi_yw_base_counts, device=device)
        L[:, i] = torch.tensor(pi_yl_attr, device=device) / torch.tensor(pi_yl_attr_counts, device=device) - torch.tensor(pi_yl_base, device=device) / torch.tensor(pi_yl_base_counts, device=device)

    return W-L

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

def drift_score_bon_batched(data, pi, tokenizer, s0: str, s_list: list[str], p, device, batch_size=8, batch_size_outer=8):
    """
    Batched version of drift_score_bon. Processes multiple (question, outputs) pairs at once for efficiency.
    Args:
        data: list of (question, outputs) pairs
        pi: model
        tokenizer: tokenizer
        s0: base system prompt
        s_list: list of persona prompts
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