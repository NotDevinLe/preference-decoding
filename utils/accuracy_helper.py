import numpy as np
import json
import argparse
from attribute_prompts import attribute_prompts
from drift import get_log_probs
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1")
args = parser.parse_args()

with open(f"../data/toy/{args.name}_test.json", "r") as f:
    data = json.load(f)
data = data[:500]

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prompt_list = [d['prompt'] for d in data]
chosen_list = [d['chosen'] for d in data]
rejected_list = [d['rejected'] for d in data]

#  Set attribute prompts here
attributes = attribute_prompts

W = np.zeros((len(data), len(attributes)))
L = np.zeros((len(data), len(attributes)))

base_prompt = "You are an AI assistant."

base_chosen, base_chosen_counts = get_log_probs(model, tokenizer, [base_prompt] * len(prompt_list), prompt_list, chosen_list, device, temperature=0.0)
base_rejected, base_rejected_counts = get_log_probs(model, tokenizer, [base_prompt] * len(prompt_list), prompt_list, rejected_list, device, temperature=0.0)

base_chosen = np.array(base_chosen) / np.array(base_chosen_counts)
base_rejected = np.array(base_rejected) / np.array(base_rejected_counts)

for i, system_prompt in tqdm.tqdm(enumerate(attributes)):
    chosen_logprobs, chosen_counts = get_log_probs(model, tokenizer, [system_prompt] * len(prompt_list), prompt_list, chosen_list, device, temperature=0.0)
    rejected_logprobs, rejected_counts = get_log_probs(model, tokenizer, [system_prompt] * len(prompt_list), prompt_list, rejected_list, device, temperature=0.0)

    W[:, i] = np.array(chosen_logprobs) / np.array(chosen_counts) - base_chosen
    L[:, i] = np.array(rejected_logprobs) / np.array(rejected_counts) - base_rejected

full = W - L

with open(f"../results/training_matrix/{args.name}.json", "w") as f:
    json.dump(full.tolist(), f)
