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

def log_prob(pi, y, system, question, device, tokenizer):
  """
  1. Do a log_softmax over the vocab dimension since we want log(pi) and not just pi
  2. We want to only compute for the response and not the prompts
  3. Sum over it
  """

  with torch.no_grad():
    full_messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": y.lstrip()}
    ]

    prompt_messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": question.strip()}
    ]

    input_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    encoded_input = tokenizer(input_text, return_tensors="pt").to(device)
    encoded_prompt = tokenizer(prompt_text, return_tensors="pt").to(device)

    input_ids = encoded_input.input_ids
    prompt_length = encoded_prompt.input_ids.shape[1]

    output_ids = input_ids[0, prompt_length:]  # (PT)
    output_logits = pi(input_ids).logits[0, prompt_length:]  # (PT, V)

    probs = F.log_softmax(output_logits, dim=-1)  # (PT, V)
    log_prob = probs.gather(dim=1, index=output_ids.unsqueeze(-1)).squeeze(-1).sum()

    return log_prob


def approximate(
    data: list[tuple[str, str, str]],
    pi, # The small model
    tokenizer,
    s0: str,
    s: list[str],
    device):

  pi.eval()
  m, k = len(data), len(s)
  W, L = torch.zeros(m, k, device=device), torch.zeros(m, k, device=device)

  for j, (question, yw, yl) in enumerate(data):
    print(j)
    for i, system in enumerate(s):
      pi_yw_attribute = log_prob(pi, yw, system, question, device, tokenizer)
      pi_yl_attribute = log_prob(pi, yl, system, question, device, tokenizer)
      pi_yw_base = log_prob(pi, yw, s0, question, device, tokenizer)
      pi_yl_base = log_prob(pi, yl, s0, question, device, tokenizer)

      W[j, i] = pi_yw_attribute - pi_yw_base
      L[j, i] = pi_yl_attribute - pi_yl_base

  d = torch.sum(W - L, dim=0)
  p = d / torch.norm(d, p=2) # (k)

  # The max direction with a unit vector is a vector that points in the same direction
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

def get_approximation_accuracy(data, p, attribute_prompts, model_ds):
  correct = 0

  for i, (question, yw, yl) in enumerate(data):
    if i % 10 == 0:
        print(f"{i}th training point")
    score = 0
    yw_base = log_prob(model_ds, yw, base_prompt, question, device)
    yl_base = log_prob(model_ds, yl, base_prompt, question, device)
    
    for pi, system_prompt in zip(p, attribute_prompts):
      yw_attribute = log_prob(model_ds, yw, system_prompt, question, device)
      yl_attribute = log_prob(model_ds, yl, system_prompt, question, device)
      score += pi * ((yw_attribute - yw_base) - (yl_attribute - yl_base))

    if score > 0:
      correct += 1

  return correct / len(data)
