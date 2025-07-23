import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
# Disable TorchDynamo to avoid disk quota issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Additional environment variables to reduce disk usage
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import pickle
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
from personas import personas
import json
from openai import OpenAI
import time
from dotenv import load_dotenv

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this checkpoint (number of completed items)")
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Generate train or test split")
args = parser.parse_args()

sample_size = args.sample_size

# Load environment variables from .env file
load_dotenv(dotenv_path="/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/.env")

# Set up OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables")
client = OpenAI(api_key=api_key)

# Model setup
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    dtype="float16",
    tensor_parallel_size=1,
    trust_remote_code=True
)

# Sampling configuration
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    max_tokens=512,
    stop=[]
)

def build_prompt(instruction, context):
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction

def create_judgment_prompt(instruction, response_1, response_2, system_prompt):
    return f"""You are an expert evaluator. Given an instruction and two responses, determine which response better follows the given system prompt.

System Prompt: {system_prompt}

Instruction: {instruction}

Response A: {response_1}

Response B: {response_2}

Which response better follows the system prompt? Respond with only "A" or "B"."""

def get_gpt4o_judgment(prompt, max_retries=3):
    """Get judgment from GPT-4o with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            judgment = response.choices[0].message.content.strip().upper()
            # Validate judgment
            if judgment in ["A", "B"]:
                return judgment
            else:
                print(f"Invalid judgment received: {judgment}, retrying...")
                time.sleep(1)
        except Exception as e:
            print(f"Error calling GPT-4o (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached, defaulting to A")
                return "A"
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    return "A"  # Default fallback

def check_checkpoint_status(checkpoint_file):
    """Check the status of the latest checkpoint"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            completed = checkpoint_data.get("completed_items", 0)
            total = checkpoint_data.get("total_items", 0)
            last_updated = checkpoint_data.get("last_updated", "Unknown")
            print(f"Found checkpoint: {completed}/{total} items completed (last updated: {last_updated})")
            return completed
    return 0

# Load Dolly dataset
dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# Select persona prompt for the user
persona_prompt = None
with open("../../data/user_prompts.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if data["user"] == args.name:
            persona_prompt = data["prompt"]
            break

assert persona_prompt is not None, f"Persona prompt for {args.name} not found"

# Use the persona prompt as the base prompt for generation
base_prompt = "You are a helpful AI assistant."

# Prepare prompts
instructions = [build_prompt(row["instruction"], row["context"]) for row in dolly_ds.shuffle().select(range(sample_size))]

# Checkpoint functionality
checkpoint_dir = f"../../data/preference/checkpoints/{args.name}"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file = f"{checkpoint_dir}/checkpoint_{args.split}.json"

# Load existing data if resuming
data = []
if args.resume_from > 0:
    completed_items = check_checkpoint_status(checkpoint_file)
    if completed_items > 0:
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            data = checkpoint_data.get("data", [])
        args.resume_from = completed_items
    else:
        print(f"Warning: Resume requested but no checkpoint found. Starting from beginning.")
        args.resume_from = 0
else:
    # Check if there's an existing checkpoint even if not explicitly resuming
    completed_items = check_checkpoint_status(checkpoint_file)
    if completed_items > 0:
        print(f"Found existing checkpoint with {completed_items} items. Use --resume_from {completed_items} to resume.")
        print("Starting fresh generation...")

# Calculate starting point
start_idx = args.resume_from
print(f"Starting {args.split} split generation from item {start_idx} out of {sample_size}")

# Batch generation
batch_size = 256
checkpoint_interval = 100  # Save checkpoint every 100 items

for i in range(start_idx, len(instructions), batch_size):
    items_remaining = len(instructions) - i
    print(f"Processing batch starting at item {i} ({items_remaining} items remaining)")
    batch = instructions[i:i + batch_size]

    inputs = []

    for instr in batch:
        # Add two inputs per instruction - both using the same base prompt
        # This will generate two different responses due to randomness in sampling
        inputs.append(tokenizer.apply_chat_template([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

        inputs.append(tokenizer.apply_chat_template([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(inputs, sampling_params)

    # Process outputs and prepare for judgment
    batch_data = []
    
    for j in range(0, len(batch)):
        output_1 = outputs[j * 2].outputs[0].text.strip()
        output_2 = outputs[j * 2 + 1].outputs[0].text.strip()
        
        batch_data.append({
            "prompt": batch[j], 
            "answer_1": output_1, 
            "answer_2": output_2,
        })
    
    # Get judgments from GPT-4o
    print(f"  Getting GPT-4o judgments for {len(batch_data)} items...")
    judgments = []
    for j in range(0, len(batch_data)):
        prompt_for_judge = create_judgment_prompt(batch_data[j]["prompt"], batch_data[j]["answer_1"], batch_data[j]["answer_2"], persona_prompt)
        judgment = get_gpt4o_judgment(prompt_for_judge)
        judgments.append(judgment)
        if j % 10 == 0:  # Print progress every 10 items
            print(f"    Judged {j+1}/{len(batch_data)} items")
    
    print(f"  Completed judgments for batch {i//batch_size + 1}")
    
    # Add judgments and assign chosen/rejected based on preference
    for j, judgment in enumerate(judgments):
        
        # Assign chosen and rejected based on judge's preference
        if judgment == "A":
            chosen = batch_data[j]["answer_1"]
            rejected = batch_data[j]["answer_2"]
        elif judgment == "B":
            chosen = batch_data[j]["answer_2"]
            rejected = batch_data[j]["answer_1"]
        else:
            # If judge gives unclear response, default to answer_1 as chosen
            chosen = batch_data[j]["answer_1"]
            rejected = batch_data[j]["answer_2"]
            judgment = "A"  # Default judgment
        
        # Create the final data point in the desired format
        data.append({
            "prompt": batch_data[j]["prompt"],
            "chosen": chosen,
            "rejected": rejected
        })
    
    # Save checkpoint every 100 items
    if len(data) % checkpoint_interval == 0:
        checkpoint_data = {
            "data": data,
            "completed_items": len(data),
            "total_items": sample_size,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  Checkpoint saved: {len(data)} items completed")

# Save final checkpoint and dataset
checkpoint_data = {
    "data": data,
    "completed_items": len(data),
    "total_items": sample_size,
    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "status": "completed"
}
with open(checkpoint_file, "w") as f:
    json.dump(checkpoint_data, f, indent=2)

# Save final dataset
os.makedirs("../../data/preference", exist_ok=True)
output_file = f"../../data/preference/{args.name}_{args.split}.json"
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nGeneration complete!")
print(f"Generated {len(data)} preference pairs for user: {args.name} ({args.split} split)")
print(f"Dataset saved to: {output_file}")
print(f"Final checkpoint saved to: {checkpoint_file}")
print(f"Note: Used LLaMA for generation and GPT-4o for judging to optimize cost and quality")

