import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
# Disable TorchDynamo to avoid disk quota issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Additional environment variables to reduce disk usage
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import time

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Generate train or test split")
args = parser.parse_args()

sample_size = args.sample_size


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

def build_prompt(instruction, context):
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction

# Prepare prompts
instructions = [build_prompt(row["instruction"], row["context"]) for row in dolly_ds.shuffle().select(range(sample_size))]

# Batch generation
batch_size = 256

all_data = []  # Accumulate all preference data here

for i in range(0, len(instructions), batch_size):
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
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(inputs, sampling_params)

    # Process outputs and prepare for judgment
    for j in range(0, len(batch)):
        # base prompt output (rejected)
        rejected = outputs[j * 2].outputs[0].text.strip()
        # attribute/persona prompt output (chosen)
        chosen = outputs[j * 2 + 1].outputs[0].text.strip()
        
        all_data.append({
            "prompt": batch[j],
            "chosen": chosen,
            "rejected": rejected,
        })


# Save final dataset
os.makedirs("../../data/preference", exist_ok=True)
output_file = f"../../data/preference/{args.name}_{args.split}.json"
with open(output_file, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\nGeneration complete!")
print(f"Generated {len(all_data)} preference pairs for user: {args.name} ({args.split} split)")
print(f"Dataset saved to: {output_file}")