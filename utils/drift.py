import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"
big_model_id   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_ds = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model_bs = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)

def log_prob(pi, y, system, question, device):
  """

  1. Do a log_softmax over the vocab dimension since we want log(pi) and not just pi
  2. We want to only compute for the response and not the prompts
  3. Sum over it

  """

  with torch.no_grad():
    input = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + system.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>user<|end_header_id|>\n"
        + question.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
        + y.lstrip()
    )

    prompt_only = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + system.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>user<|end_header_id|>\n"
        + question.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    encoded_input = tokenizer(input, return_tensors="pt").to(device)
    encoded_prompt = tokenizer(prompt_only, return_tensors="pt").to(device)

    input_ids = encoded_input.input_ids
    prompt_length = encoded_prompt.input_ids.shape[1]

    output_ids = input_ids[0, prompt_length:] # (PT)
    output_logits = pi(input_ids).logits[0, prompt_length:] # (PT, V)

    probs =  F.log_softmax(output_logits, dim=-1) # (PT, V)

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
    for i, system in enumerate(s):

      pi_yw_attribute = log_prob(pi, yw, system, question, device)
      pi_yl_attribute = log_prob(pi, yl, system, question, device)
      pi_yw_base = log_prob(pi, yw, s0, question, device)
      pi_yl_base = log_prob(pi, yl, s0, question, device)

      W[j, i] = pi_yw_attribute - pi_yw_base
      L[j, i] = pi_yl_attribute - pi_yl_base

  d = torch.sum(W - L, dim=0)
  p = d / torch.norm(d, p=2) # (k)

  # The max direction with a unit vector is a vector that points in the same direction
  return p

def get_logits(pi, y, system, question, device):
  """

  Same logic as the log_prob method except now you're not summing
  over anything at the end, you just grab the last time step

  """

  with torch.no_grad():
    prompt_only = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + system.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>user<|end_header_id|>\n"
        + question.strip() + "<|eot_id|>\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    encoded_prompt = tokenizer(prompt_only, return_tensors="pt").to(device)
    prompt_ids = encoded_prompt.input_ids
    input_ids = prompt_ids

    if len(y) != 0:
      gen_ids   = torch.tensor([y], device=device)
      input_ids = torch.cat([prompt_ids, gen_ids], dim=1)

    output_logits = pi(input_ids).logits[0, -1, :] # (V)

    return output_logits

def decode(
    x: str,
    large_llm,
    small_llm,
    tokenizer,
    s0: str,
    s: list[str],
    p: list[float],
    b: float,
    max_tokens: int,
    device):

  y = [] # This will contain the token ids

  """

  For each time step, we want to get the logits given x + y for our big and small model
  Repeat this for all k attributes
  Calculate drift logits and then sample from that logit distribution
  Append that token into y and repeat

  """

  for i in range(max_tokens):
    if i % 10 == 0:
      print(f"Token: {i}")

    h_large = get_logits(large_llm, y, s0, x, device)
    h_base = get_logits(small_llm, y, s0, x, device)

    for pi, si in zip(p, s):
      h_ti = get_logits(small_llm, y, si, x, device)
      h_large += pi * (h_ti - h_base) / b

    h_large = F.softmax(h_large, dim=-1)
    token_id = torch.multinomial(h_large, num_samples=1).item()
    y.append(token_id)

    if token_id == tokenizer.eos_token_id:
      break

  return tokenizer.decode(y, skip_special_tokens=True)
