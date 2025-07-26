import argparse
import json
import torch
import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
from drift import approximate, approximate_l1
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from attribute_prompts import attribute_prompts
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
parser.add_argument('--save_path', type=str, default="../results/user_p.jsonl", help='Path to save results')
parser.add_argument('--reg_type', type=str, choices=['l2', 'l1'], default='l1', help='Type of regularization: l2 or l1')
parser.add_argument('--lambda_reg', type=float, default=0.1, help='L1 regularization strength (only used if reg_type=l1)')
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

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

engine = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.95, max_model_len=4096)

# Get base prompt (system prompt used for base)
base_prompt = "You are an AI assistant."

print("Computing drift approximation vector p...")
# Compute drift approximation vector p

sample_sizes = [10]

for sample_size in sample_sizes:
    # Convert to format expected by approximate function: (question, yw, yl)
    # where question is the prompt, yw is chosen response, yl is rejected response
    folds = []
    random.shuffle(preference_data)

    for i in range(0, sample_size, sample_size):
        current_fold = []
        for j in range(i, i + sample_size):
            question = preference_data[j]['prompt']
            yw = preference_data[j]['chosen']  # winning/chosen response
            yl = preference_data[j]['rejected']  # losing/rejected response
            current_fold.append((question, yw, yl))
        folds.append(current_fold)

    print(f"Converted {len(folds)} samples to drift format")

    results = []

    for fold in folds:
        if args.reg_type == 'l2':
            p = approximate(fold, engine, tokenizer, base_prompt, attribute_prompts, device, batch_size=8)
        elif args.reg_type == 'l1':
            p = approximate_l1(fold, engine, tokenizer, base_prompt, attribute_prompts, device, lambda_reg=args.lambda_reg)
        else:
            raise ValueError(f"Unknown reg_type: {args.reg_type}")
        results.append(p.tolist())

    # Save p to jsonl
    result_entry = {
        "user": args.name,
        "n": sample_size,
        "lambda_reg": args.lambda_reg,
        "p": results,
    }

    with open(args.save_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")