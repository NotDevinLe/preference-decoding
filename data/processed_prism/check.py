from datasets import load_dataset
import random
import json
from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def _build_full_prompt(tokenizer, sys_prompt: str, user_prompt: str, completion: str):
    """Return: full_text (prompt+completion), prefix_tokens, completion_tokens"""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_prompt.strip()},
        {"role": "user",   "content": user_prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    return prompt_text + completion, len(prompt_ids), len(comp_ids)

for i in range(25):
    dataset_path = f'data/processed_prism/user{i}.json'
    if not os.path.exists(dataset_path):
        print(f"User {i} data not found, skipping")
        continue
    with open(dataset_path, 'r') as f:
        user_data = json.load(f)
    biggest = 0
    for entry in user_data:
        prompt = entry['prompt']
        chosen = entry['chosen']
        rejected = entry['rejected']
        
        full_prompt, prefix_len, comp_len = _build_full_prompt(tokenizer, "You are a helpful assistant.", prompt, chosen)
        biggest = max(biggest, prefix_len + comp_len)
    print(f"User {i}: {biggest}")