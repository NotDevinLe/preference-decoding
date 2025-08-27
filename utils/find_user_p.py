import argparse
import json
import torch
import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
from drift import approximate
from transformers import AutoTokenizer, AutoModelForCausalLM
from attribute_prompts import attribute_prompts, base_prompt, persona_prompts, persona_prompts_2
from dotenv import load_dotenv
from huggingface_hub import login
import random
import vllm

load_dotenv(dotenv_path="/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/.env")
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not found in .env")
login(hf_token)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='User name (e.g., user1)')
parser.add_argument('--samples', type=int, default=200, help='Maximum number of samples to use')
parser.add_argument('--lambda0', type=float, default=0.01, help='Lambda0 for L1 regularization')
args = parser.parse_args()

# Load user data from JSON format
data_path = f"../data/preference/{args.name}_train.json"
print(f"Loading data from: {data_path}")

with open(data_path, "r") as f:
    preference_data = json.load(f)

print(f"Loaded {len(preference_data)} preference pairs")

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)

# model = AutoModelForCausalLM.from_pretrained(small_model_id, device_map="auto", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

# selected_attr_idx = [0, 1, 2, 31, 33, 37, 43]
# attribute_prompts = [attribute_prompts[i] for i in selected_attr_idx]

data = []
for j in range(args.samples):
    question = preference_data[j]['prompt']
    yw = preference_data[j]['chosen']  # winning/chosen response
    yl = preference_data[j]['rejected']  # losing/rejected response
    data.append((question, yw, yl))

print(f"Converted {len(data)} samples to drift format")

ns = [args.samples]
for n in ns:
    current_data = data[:n]
    p = approximate(current_data, model, tokenizer, base_prompt, attribute_prompts,l1_lambda=0.01, device=device)

    # Save p to jsonl
    result_entry = {
        "user": args.name,
        "n": n,
        "p": p.tolist(),
        "lambda0": args.lambda0
    }

    with open(f'../results/{args.name}_p.jsonl', "a") as f:
        f.write(json.dumps(result_entry) + "\n")