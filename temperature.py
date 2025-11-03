import numpy as np
import asyncio
import aiohttp
from src.core.drift import compute_rewards
from transformers import AutoTokenizer
import json

base_prompt = "You are a helpful assistant."
system_prompts = ["You are a chinese girl named Yang.", "You are an American boy named Jerry."]
data1 = json.load(open("data/preference_toy/user0_train.json"))
data2 = json.load(open("data/preference_toy/user1_train.json"))

base_model_outputs = []

async def main():
    async with aiohttp.ClientSession() as session:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        await compute_rewards(data1, 0, system_prompts, base_prompt, tokenizer, "http://localhost:8080", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        await compute_rewards(data2, 1, system_prompts, base_prompt, tokenizer, "http://localhost:8080", "meta-llama/Meta-Llama-3.1-8B-Instruct")

if __name__ == "__main__":
    asyncio.run(main())