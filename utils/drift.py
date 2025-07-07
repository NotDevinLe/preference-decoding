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

def log_prob(pi, ys, systems, questions, device, tokenizer, batch_size=8):
    log_probs = []
    pi.eval()

    with torch.no_grad():
        for i in range(0, len(ys), batch_size):
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
            prompt_lens = (encoded_prompt.attention_mask.sum(dim=1)).tolist()

            outputs = pi(input_ids).logits
            probs = F.log_softmax(outputs, dim=-1)

            for j in range(len(batch_ys)):
                start = prompt_lens[j]
                target_ids = input_ids[j, start:]
                token_probs = probs[j, start:].gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                log_probs.append(token_probs.sum().item())

    return log_probs

def approximate(data, pi, tokenizer, s0: str, s_list: list[str], device, batch_size=8):
    pi.eval()
    m, k = len(data), len(s_list)
    W = torch.zeros(m, k, device=device)
    L = torch.zeros(m, k, device=device)

    for i, system in enumerate(s_list):
        questions = [q for q, _, _ in data]
        yw_list = [yw for _, yw, _ in data]
        yl_list = [yl for _, _, yl in data]

        # Log-probabilities in batches
        pi_yw_attr = log_prob(pi, yw_list, [system]*m, questions, device, tokenizer, batch_size)
        pi_yl_attr = log_prob(pi, yl_list, [system]*m, questions, device, tokenizer, batch_size)
        pi_yw_base = log_prob(pi, yw_list, [s0]*m, questions, device, tokenizer, batch_size)
        pi_yl_base = log_prob(pi, yl_list, [s0]*m, questions, device, tokenizer, batch_size)

        W[:, i] = torch.tensor(pi_yw_attr) - torch.tensor(pi_yw_base)
        L[:, i] = torch.tensor(pi_yl_attr) - torch.tensor(pi_yl_base)

    d = torch.sum(W - L, dim=0)
    p = d / torch.norm(d, p=2)
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
            hi_small = self.get_small_logits(input_ids, attr_prompt)
            drift += w * (hi_small - h0_small)

        return aligned_logits + drift / self.b

def get_approximation_accuracy(data, model_ds, p, base_prompt, attribute_prompts, device, tokenizer, batch_size=8):
    correct = 0
    model_ds.eval()

    questions = [q for q, _, _ in data]
    yw_list = [yw for _, yw, _ in data]
    yl_list = [yl for _, _, yl in data]

    # Precompute base log-probs in batches
    yw_base_probs = log_prob(model_ds, yw_list, [base_prompt] * len(data), questions, device, tokenizer, batch_size)
    yl_base_probs = log_prob(model_ds, yl_list, [base_prompt] * len(data), questions, device, tokenizer, batch_size)

    # Precompute attribute log-probs in batches
    attr_scores = torch.zeros(len(data), len(attribute_prompts), device=device)

    for i, prompt in enumerate(attribute_prompts):
        yw_attr_probs = log_prob(model_ds, yw_list, [prompt] * len(data), questions, device, tokenizer, batch_size)
        yl_attr_probs = log_prob(model_ds, yl_list, [prompt] * len(data), questions, device, tokenizer, batch_size)

        attr_scores[:, i] = torch.tensor([
            pi * ((yw_attr - yw_base) - (yl_attr - yl_base))
            for pi, yw_attr, yl_attr, yw_base, yl_base in zip(
                [p[i]] * len(data), yw_attr_probs, yl_attr_probs, yw_base_probs, yl_base_probs
            )
        ])

    total_scores = attr_scores.sum(dim=1)
    correct = (total_scores > 0).sum().item()

    return correct / len(data)

