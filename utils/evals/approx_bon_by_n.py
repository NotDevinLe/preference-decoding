import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM
sys.path.append("LLaMA-Factory/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.drift import get_scores
    from utils.attribute_prompts import attribute_prompts, base_prompt
except ImportError:
    sys.path.append('..')
    from drift import get_scores
    from attribute_prompts import attribute_prompts, base_prompt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="user1")
parser.add_argument("--gold_model_path", type=str, required=True, help="Path to gold reward model")
args = parser.parse_args()

# Load bon outputs
data_path = "../../data/bon.json"
with open(data_path, "r") as f:
    bon_data = json.load(f)

print(f"Loaded {len(bon_data)} prompts from {data_path}")

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model_path = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize tokenizer first
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Initialize model (fixed parameter order)
model = LLM(
    model=base_model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=8192,
    dtype="bfloat16",
    trust_remote_code=True,
)

# Load gold reward model
print(f"Loading gold reward model from: {args.gold_model_path}")
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

gold_model_args = ModelArguments(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" ,
    adapter_name_or_path=args.gold_model_path,
    trust_remote_code=True,
    use_fast_tokenizer=True,
)
gold_finetuning_args = FinetuningArguments(stage="rm")

gold_tokenizer_module = load_tokenizer(gold_model_args)
gold_tokenizer = gold_tokenizer_module["tokenizer"]
if gold_tokenizer.pad_token is None:
    gold_tokenizer.pad_token = gold_tokenizer.eos_token

gold_model = load_model(
    tokenizer=gold_tokenizer,
    model_args=gold_model_args,
    finetuning_args=gold_finetuning_args,
    is_trainable=False,
    add_valuehead=True
)
gold_model.to(device)
gold_model.eval()

def format_llama3_prompt(prompt: str, response: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
    )

def get_gold_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        logits, _, values = model(**inputs)
        return values[:, -1]

# # Load p vector from JSON file
# with open(args.p_path, "r") as f:
#     data = json.load(f)

# p = None
# if isinstance(data, list):
#     # It's a list of entries
#     for entry in data:
#         if entry.get('n', entry.get('sample_size')) == 200:
#             p = np.array(entry["p"])
#             break
# else:
#     # It's a single entry
#     if data.get('n', data.get('sample_size')) == 200:
#         p = np.array(data["p"])

# if p is None:
#     raise ValueError(f"No p vector found for user {args.name} with n=200 in {args.p_path}")

# print(f"Loaded p vector for user {args.name}")

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

# Sparsify p using torch operations for consistency
p_tensor = torch.tensor(p, device=device)
abs_p = torch.abs(p_tensor)
topk_values, topk_idx = torch.topk(abs_p, k=min(7, len(p)))  # Handle case where p has < 7 elements

p_sparse = torch.zeros_like(p_tensor)
p_sparse[topk_idx] = p_tensor[topk_idx]
p_sparse_np = p_sparse.cpu().numpy()

print(f"Sparsified p: kept top {len(topk_idx)} elements")
print(f"Non-zero elements: {torch.count_nonzero(p_sparse).item()}")

# Choose the right attribute prompts
attribute_list = attribute_prompts  # or attribute_prompts, depending on what you want

print(f"Using {len(attribute_list)} attribute prompts")

# Precompute all scores for all outputs (up to max_k)
max_k = 20
print("Computing drift scores for all outputs...")
all_prompt_outputs = [(item["prompt"], item["outputs"][:20]) for item in bon_data]
all_scores_full = get_scores(
    all_prompt_outputs,
    model, p_sparse_np, base_prompt, attribute_list, device, tokenizer
)

print(f"Computed scores for {len(all_scores_full)} prompts")

# Evaluate for different values of k
results_by_k = []
for k in range(2, max_k + 1, 2):
    print(f"Evaluating for k={k}...")
    selected_gold_scores = []
    
    for item, scores in zip(bon_data, all_scores_full):
        # Use only the first k outputs and their scores
        outputs = item["outputs"][:k]
        scores_k = scores[:k]
        
        # Find best output according to our model (fix CUDA tensor issue)
        if isinstance(scores_k, torch.Tensor):
            idx = int(torch.argmax(scores_k).cpu())  # Use torch.argmax and move to CPU
        else:
            idx = int(np.argmax(scores_k))  # Use numpy if it's already a numpy array
        
        # Score the selected output with gold model
        selected_output = outputs[idx]
        formatted_selected = format_llama3_prompt(item["prompt"], selected_output)
        gold_score_selected = get_gold_score(gold_model, gold_tokenizer, formatted_selected)[0].item()
        
        selected_gold_scores.append(gold_score_selected)
    
    # Calculate average selected gold score
    avg_selected = float(np.mean(selected_gold_scores))
    
    results_by_k.append({
        "user": args.name,
        "k": k,
        "avg_selected_gold": avg_selected,
    })
    
    print(f"k={k}: avg_selected_gold={avg_selected:.4f}")

# Save results
output_file = f"../../results/mle_bon_by_n.jsonl"
with open(output_file, "a") as f:
    for res in results_by_k:
        f.write(json.dumps(res) + "\n")
print(f"✅ Results saved to {output_file}")

# Print summary
best_k = max(results_by_k, key=lambda x: x['avg_selected_gold'])
print(f"\nBest performance at k={best_k['k']} with avg_selected_gold={best_k['avg_selected_gold']:.4f}")