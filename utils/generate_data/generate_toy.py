import os
import argparse
from transformers import AutoTokenizer
import json
import asyncio
import aiohttp
import time

parser = argparse.ArgumentParser()
parser.add_argument("--name_idx", type=int, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Generate train or test split")
args = parser.parse_args()

sample_size = args.sample_size

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8080/v1"
VLLM_GENERATE_ENDPOINT = f"{VLLM_SERVER_URL}/completions"

# Generation parameters
GENERATION_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop": []
}

# Simple prompts from temperature.py
base_prompt = "You are a helpful assistant."
system_prompts = ["You are a chinese girl named Yang.", "You are an American boy named Jerry."]
questions = ["What is your name?", "Where are you from?"] * 10

# Use the specified system prompt
persona_prompt = system_prompts[args.name_idx % len(system_prompts)]
print(f"Using persona prompt {args.name_idx}: {persona_prompt}")


async def generate_text_async(session, prompt, max_retries=3):
    """Generate text using vLLM server via async HTTP request"""
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": prompt,
        **GENERATION_PARAMS
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(VLLM_GENERATE_ENDPOINT, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0].get("text", "").strip()
                else:
                    print(f"HTTP error {response.status}: {await response.text()}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
    
    print(f"Failed to generate text after {max_retries} attempts")
    return ""

async def generate_batch_async(questions, base_prompt, persona_prompt):
    """Generate responses for a batch of questions"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for question in questions:
            # Create prompts for both base and persona
            base_input = tokenizer.apply_chat_template([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
            
            persona_input = tokenizer.apply_chat_template([
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
            
            # Create async tasks for both generations
            tasks.append(generate_text_async(session, base_input))
            tasks.append(generate_text_async(session, persona_input))
        
        # Wait for all generations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_data = []
        for i in range(0, len(results), 2):
            if i + 1 < len(results):
                output_1 = results[i] if not isinstance(results[i], Exception) else ""
                output_2 = results[i + 1] if not isinstance(results[i + 1], Exception) else ""
                
                batch_data.append({
                    "prompt": questions[i // 2],
                    "chosen": output_2,
                    "rejected": output_1
                })
        
        return batch_data

# Use the questions from temperature.py, repeat as needed based on sample_size
instructions = (questions * ((sample_size // len(questions)) + 1))[:sample_size]

async def main():
    data = []
    batch_size = 2048  # Smaller batch size for async requests
    total_batches = (len(instructions) + batch_size - 1) // batch_size
    
    for i in range(0, len(instructions), batch_size):
        items_remaining = len(instructions) - i
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} starting at item {i} ({items_remaining} items remaining)")
        
        batch = instructions[i:i + batch_size]
        
        # Generate batch asynchronously
        batch_data = await generate_batch_async(batch, base_prompt, persona_prompt)
        data.extend(batch_data)
        
        print(f"Completed batch {batch_num}/{total_batches}")
    
    return data

# Run the async main function
data = asyncio.run(main())


# Save final dataset
os.makedirs("data/preference_toy", exist_ok=True)
output_file = f"data/preference_toy/user{args.name_idx}_{args.split}.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nGeneration complete!")
print(f"Generated {len(data)} preference pairs for user: {args.name_idx} ({args.split} split)")
print(f"Dataset saved to: {output_file}")