import os
import argparse
from transformers import AutoTokenizer
import json
import asyncio
import aiohttp
import time

parser = argparse.ArgumentParser()
parser.add_argument("--name_idx_range", type=str, required=True, help="Comma-separated list of user indices, e.g., '0,1,2' or '0-2'")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Generate train or test split")
args = parser.parse_args()

# Parse name_idx_range
if '-' in args.name_idx_range:
    # Handle range format like "0-2"
    start, end = map(int, args.name_idx_range.split('-'))
    name_indices = list(range(start, end + 1))
else:
    # Handle comma-separated format like "0,1,2"
    name_indices = [int(x.strip()) for x in args.name_idx_range.split(',')]

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8080/v1"
VLLM_GENERATE_ENDPOINT = f"{VLLM_SERVER_URL}/completions"

# Generation parameters
GENERATION_PARAMS = {
    "temperature": 0,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop": []
}

# Load configuration data
with open("configs/persona_prompts.json", "r") as f:
    persona_data = json.load(f)

with open("configs/high_variance_questions.json", "r") as f:
    questions = json.load(f)

# Validate all name indices
for name_idx in name_indices:
    if name_idx >= len(persona_data["prompts"]):
        raise ValueError(f"Persona prompt index {name_idx} is out of range. Available indices: 0-{len(persona_data['prompts'])-1}")

base_prompt = "You are a helpful AI assistant."
instructions = questions["test"]

async def generate_text_async(session, prompt, max_retries=3):
    """Generate text using vLLM server via async HTTP request"""
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
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

async def generate_batch_async(instructions, base_prompt, persona_prompt):
    """Generate responses for a batch of instructions"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for instr in instructions:
            # Create prompts for both base and persona
            base_input = tokenizer.apply_chat_template([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": instr}
            ], tokenize=False, add_generation_prompt=True)
            
            persona_input = tokenizer.apply_chat_template([
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": instr}
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
                    "prompt": instructions[i // 2],
                    "chosen": output_2,
                    "rejected": output_1
                })
        
        return batch_data

def apply_persona_template(persona_prompt):
    template = f"""
    
    You are roleplaying as a real person with the following characteristics and background:

    {persona_prompt}

    CRITICAL INSTRUCTIONS:
    - Respond naturally as this person would, drawing on their life experience, personality, and worldview
    - Let your age, education, occupation, and personality traits authentically shape your perspectives and communication style
    - Never explicitly mention your demographic details, scores, or background unless directly relevant to the conversation
    - Don't say things like "as someone with high agreeableness" or "given my neuroticism score"
    - Simply BE this person - your responses should reflect these traits implicitly through your opinions, word choices, knowledge depth, and reasoning

    Your responses should naturally reflect:
    - How someone with your education and occupation would explain concepts
    - How your personality traits influence your communication style and viewpoints
    - What life experiences someone of your age and background would reference
    - The perspective someone with your ideology and values would take
    - The interests and knowledge areas relevant to your quirks and lifestyle

    Respond authentically as this person would in everyday conversation.
    """
    return template.format(persona_prompt=persona_prompt)

async def generate_for_user(name_idx, persona_prompt):
    """Generate data for a single user"""
    print(f"\n=== Processing user {name_idx} ===")
    print(f"Using persona prompt {name_idx}: {persona_prompt[:100]}...")
    
    data = []
    batch_size = 2048  # Smaller batch size for async requests
    total_batches = (len(instructions) + batch_size - 1) // batch_size
    
    for i in range(0, len(instructions), batch_size):
        items_remaining = len(instructions) - i
        batch_num = i // batch_size + 1
        print(f"User {name_idx}: Processing batch {batch_num}/{total_batches} starting at item {i} ({items_remaining} items remaining)")
        
        batch = instructions[i:i + batch_size]
        
        # Generate batch asynchronously
        batch_data = await generate_batch_async(batch, base_prompt, persona_prompt)
        data.extend(batch_data)
        
        print(f"User {name_idx}: Completed batch {batch_num}/{total_batches}")
    
    return data

async def main():
    """Generate data for all users in the range"""
    all_results = {}
    
    for name_idx in name_indices:
        persona_prompt = persona_data["prompts"][name_idx]
        formatted_persona = apply_persona_template(persona_prompt)
        data = await generate_for_user(name_idx, formatted_persona)
        all_results[name_idx] = data
    
    return all_results

# Run the async main function
all_data = asyncio.run(main())

# Save datasets for each user
os.makedirs("data/PERSONA_testing", exist_ok=True)
total_pairs = 0

for name_idx, data in all_data.items():
    output_file = f"data/PERSONA_testing/user{name_idx}_{args.split}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"User {name_idx}: Generated {len(data)} preference pairs")
    print(f"User {name_idx}: Dataset saved to: {output_file}")
    total_pairs += len(data)

print(f"\n=== Generation complete! ===")
print(f"Generated {total_pairs} total preference pairs across {len(name_indices)} users")
print(f"Users processed: {name_indices}")