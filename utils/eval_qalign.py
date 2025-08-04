from qalign import qalign
import torch
import numpy as np
import random
from vllm import LLM, SamplingParams
from typing import List, Tuple, Optional
import math
from transformers import AutoTokenizer
from attribute_prompts import base_prompt
import sys
import json
import argparse

# Add LLaMA-Factory to path
sys.path.append("LLaMA-Factory/src")

from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import ModelArguments, FinetuningArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="user1")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = LLM(
        model=model_id,
        tokenizer=model_id,
        max_model_len=2048,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_path = f"/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/utils/saves/normal/user1_1b/toy_reward_200"

    print("Loading model...")
    # Setup arguments for LLaMA-Factory
    model_args = ModelArguments(
        model_name_or_path=model_id,
        adapter_name_or_path=adapter_path,
        trust_remote_code=True,
        use_fast_tokenizer=True,
    )

    finetuning_args = FinetuningArguments(
        stage="rm"
    )

    # Load tokenizer and model properly using LLaMA-Factory
    print("Loading tokenizer...")
    tokenizer_module = load_tokenizer(model_args)
    reward_tokenizer = tokenizer_module["tokenizer"]

    # Set padding token
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        print(f"Set pad_token to eos_token: {reward_tokenizer.pad_token}")

    print("Loading reward model...")
    reward_model = load_model(
        tokenizer=reward_tokenizer,
        model_args=model_args,
        finetuning_args=finetuning_args,
        is_trainable=False,
        add_valuehead=True
    )

    reward_model.to(device)
    reward_model.eval()

    def format_llama3_prompt(prompt: str, response: str) -> str:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n" + response.strip() + "<|eot_id|>"
        )

    # Create a closure that captures the reward model and dependencies
    def create_reward_function(reward_model, tokenizer, device):
        def reward_fn(question, response):
            # Format the conversation using LLaMA format
            conversation_text = format_llama3_prompt(question, response)
            
            inputs = tokenizer(conversation_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                logits, _, values = reward_model(**inputs)
                # Extract the reward score
                return values[:, -1][0].item()
        return reward_fn

    # Create the reward function with captured dependencies
    reward_fn = create_reward_function(reward_model, reward_tokenizer, device)

    with open("../data/bon.json", "r") as f:
        data = json.load(f)

    questions = [item["prompt"] for item in data]

    results = []
    for question in questions:
        output = qalign(
            model=model,  # Use the vLLM model for generation
            tokenizer=tokenizer,
            question=question,
            reward_fn=reward_fn,
            base_prompt=base_prompt,
            num_steps=args.n,  # Reduced for testing
            beta=1.0,
        )
        results.append(output)
    with open(f"results/qalign_results/{args.name}_sample{args.n}.json", "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()