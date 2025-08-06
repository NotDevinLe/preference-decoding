import json
import torch
import numpy as np
import sys
from drift import get_approximation_accuracy
from attribute_prompts import attribute_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import vllm

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
# parser.add_argument("--p_path", type=str, required=True)
parser.add_argument("--k", type=int, default=7)
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

# Settings
test_path = f"../data/preference/{args.name}_test.json"
# p_path = args.p_path
base_model_path = "meta-llama/Llama-3.2-1B-Instruct"
base_prompt = "You are an AI assistant."

# Load test data
with open(test_path, "r") as f:
    test_data = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

engine = vllm.LLM(model=base_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)
# model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)

# Prepare data for get_approximation_accuracy
eval_data = [
    (entry["prompt"], entry["chosen"], entry["rejected"])
    for entry in test_data
]

print("Finished loading data")

def sparsify_p(p_list, k=14):
    p_list = np.array(p_list)
    abs_p = np.abs(p_list)
    topk_idx = np.argsort(abs_p)[-k:]
    p_sparse = np.zeros_like(p_list)
    p_sparse[topk_idx] = p_list[topk_idx]
    return p_sparse

eval_data = eval_data[:args.sample_size]

p = [
    37.69054412841797,
    -2.9021151065826416,
    -0.25404518842697144,
    3.0760273933410645,
    4.032073974609375,
    -6.471762657165527,
    -5.1571831703186035,
    -6.6313886642456055,
    7.341609001159668,
    -13.851171493530273,
    9.581570625305176,
    4.129615783691406,
    1.0026273727416992,
    11.910504341125488,
    -0.9516629576683044,
    3.01949405670166,
    -2.473126173019409,
    8.309188842773438,
    3.2338342666625977,
    -2.9333927631378174,
    5.410052299499512,
    22.96633529663086,
    -3.95829439163208,
    -8.865690231323242,
    9.30678939819336,
    4.758558750152588
  ]

p = np.array(p)
p = p / np.linalg.norm(p)
p = p.tolist()

accuracy = get_approximation_accuracy(
    eval_data,
    engine,
    sparsify_p(p, args.k),
    base_prompt,
    attribute_prompts,
    device,
    tokenizer,
    batch_size=8
)

print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(eval_data))}/{len(eval_data)})")

# Save results
with open(args.save_path, "a") as f:
    f.write(json.dumps({
        "user": args.name,
        "n": args.sample_size,
        "acc": accuracy,
        "k": args.k
    }) + "\n")
print(f"Results saved to {args.save_path}")