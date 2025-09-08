import json
import torch
import numpy as np
import sys
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import vllm

sys.path.append("..")
from drift import get_approximation_accuracy
from attribute_prompts import attribute_prompts, persona_prompts_3, attribute_selected, persona_selected

parser = argparse.ArgumentParser()
parser.add_argument("--names", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

names = args.names.split(",")

# Settings

base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
base_prompt = "You are an AI assistant."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

engine = vllm.LLM(model=base_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)

for name in names:
    print(f"Evaluating {name}")
    test_path = f"../../data/persona_pref/{name}_test.json"
    p_path = f"../../results/group1/{name}_p.jsonl"

    # Load test data
    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Prepare data for get_approximation_accuracy
    eval_data = [
        (entry["prompt"], entry["chosen"], entry["rejected"])
        for entry in test_data
    ]

    print("Finished loading data")

    def sparsify_p(p_list, k=7):
        p_list = np.array(p_list)
        abs_p = np.abs(p_list)
        topk_idx = np.argsort(abs_p)[-k:]
        p_sparse = np.zeros_like(p_list)
        p_sparse[topk_idx] = p_list[topk_idx]
        return p_sparse

    with open(p_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            p = entry["p"]
            p = np.array(p)
            p = p / np.linalg.norm(p)
            p = p.tolist()

            if entry["lambda0"] != 0:
                continue

            system_prompts = []
            selected_attr_idx = []
            if entry["system_prompt_list"] == "personas":
                system_prompts = persona_prompts_3
                selected_attr_idx = persona_selected[entry["lambda0"]]
            else:
                system_prompts = attribute_prompts
                selected_attr_idx = attribute_selected[entry["lambda0"]]
            system_prompts = [system_prompts[i] for i in selected_attr_idx]

            accuracy = get_approximation_accuracy(
                eval_data,
                engine,
                sparsify_p(p, k=7),
                base_prompt,
                system_prompts,
                device,
                tokenizer
            )

            print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(eval_data))}/{len(eval_data)})")

            # Save results
            with open(args.save_path, "a") as f:
                f.write(json.dumps({
                    "user": name,
                    "acc": accuracy,
                    "lambda_val": entry["lambda0"],
                    "system_prompt_list": entry["system_prompt_list"]
                }) + "\n")
            print(f"Results saved to {args.save_path}")
    print(f"Finished evaluating {name}")