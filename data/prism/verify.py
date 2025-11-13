from datasets import load_dataset
import random
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def _build_full_prompt_multi_turn(tokenizer, sys_prompt: str, user_prompts, completion: str):
    """Pass in the prompt value of the prism dataset for user_prompts don't change anything okay?"""

    conversation = [{"role": "system", "content": sys_prompt.strip()}]
    for prompt in user_prompts:
        conversation.append({"role": prompt["role"], "content": prompt["content"].strip()})

    prompt_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids   = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    return prompt_text + completion, len(prompt_ids), len(comp_ids)

ds = load_dataset("parquet", data_files="data/prism/train.parquet")
save_path = "data/processed_prism"

for i in range(100):
    user_data = ds['train'].filter(lambda x: x['extra_info']['user_id'] == f"user{i}")

    processed_data = []

    new_user_data = []
    for row in user_data:
        full_prompt, prefix_len, comp_len = _build_full_prompt_multi_turn(tokenizer, "You are a helpful assistant.", row['prompt'], row['extra_info']['chosen_utterance'])
        new_user_data.append({"prompt": full_prompt, "chosen": row['extra_info']['chosen_utterance'], "rejected": random.choice(row['extra_info']['rejected_utterance'])})

    if len(new_user_data) > 0:
        with open(f"{save_path}/user{i}.json", "w") as f:
            json.dump(new_user_data, f, indent=2)