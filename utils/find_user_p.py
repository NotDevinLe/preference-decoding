import argparse
import json
import torch
from drift import approximate
from transformers import AutoTokenizer
from attribute_prompts import attribute_prompts, base_prompt, persona_prompts_3, persona_selected, attribute_selected
import vllm

parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, required=True, help='Comma-separated list of user names (e.g., user1,user2)')
parser.add_argument('--samples', type=int, default=150, help='Maximum number of samples to use')
# parser.add_argument('--lambda0', type=float, default=0.01, help='Lambda0 for L1 regularization')
parser.add_argument('--system_prompt_list', type=str, choices=['personas', 'attributes'], default='personas')
args = parser.parse_args()

names = args.names.split(',')

# Model and tokenizer setup
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = vllm.LLM(model=small_model_id, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

for name in names:
    print(f"Finding p vector for {name}")
    # Load user data from JSON format
    data_path = f"../data/persona_pref/{name}_train.json"
    print(f"Loading data from: {data_path}")

    with open(data_path, "r") as f:
        preference_data = json.load(f)

    print(f"Loaded {len(preference_data)} preference pairs")

    lambdas = [0]

    for lambda_ in lambdas:
        print(f"Finding p vector for lambda={lambda_}")
        system_prompts = persona_prompts_3 if args.system_prompt_list == 'personas' else attribute_prompts
        selected_idx = persona_selected if args.system_prompt_list == 'personas' else attribute_selected
        selected_attr_idx = selected_idx[lambda_]
        system_prompts = [system_prompts[i] for i in selected_attr_idx]

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
            p = approximate(current_data, model, tokenizer, base_prompt, system_prompts,l1_lambda=0.01, device=device)

            # Save p to jsonl
            result_entry = {
                "user": name,
                "n": n,
                "p": p.tolist(),
                "lambda0": lambda_,
                "system_prompt_list": args.system_prompt_list
            }

            with open(f'../results/{name}_p.jsonl', "a") as f:
                f.write(json.dumps(result_entry) + "\n")