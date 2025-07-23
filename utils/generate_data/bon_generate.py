import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
import sys
# Append project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import pickle
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--n", type=int, required=True)
args = parser.parse_args()
sample_size = args.sample_size
n = args.n

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
    max_tokens=256,
    stop=[]
)

# Load Dolly dataset
print("Loading Dolly dataset...")
dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# Prepare prompts (no system prompt)
instructions = [row["instruction"] for row in dolly_ds.shuffle().select(range(sample_size))]

results = []

# Batch generation
batch_size = 64
for i in range(0, len(instructions), batch_size):
    print(f"Generating batch {i} to {i+batch_size}...")
    batch = instructions[i:i + batch_size]

    attr_inputs = []
    for instr in batch:
        # Only user message, no system prompt
        attr_inputs.append(tokenizer.apply_chat_template([
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

    # Generate num_outputs completions for each input in the batch
    attr_outputs = llm.generate(attr_inputs, SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=256,
        stop=[],
        n=n
    ))

    # attr_outputs is a list of objects, each with .outputs (list of num_outputs generations)
    for prompt, attr in zip(batch, attr_outputs):
        outputs = [out.text.strip() for out in attr.outputs]
        results.append({
            "prompt": prompt,
            "outputs": outputs
        })

# Save dataset as JSON
with open("../../data/bon_200.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved {len(results)} prompt generations to ../data/bon_200.json")

