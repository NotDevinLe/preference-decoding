import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
import torch
import pickle
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
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
    max_tokens=256,
    stop=[]
)

# Load Dolly dataset
dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")
persona_ds = load_dataset("kaist-ai/Multifaceted-Collection", split="train")

# Prepare prompts
instructions = [row["instruction"] for row in dolly_ds.shuffle().select(range(sample_size))]
data = []

# Batch generation
batch_size = 32
for i in range(0, len(instructions), batch_size):
    print(f"Finished batch {i}")
    batch = instructions[i:i + batch_size]
    persona_prompt = persona_ds.shuffle()[0]["system"]
    base_prompt = "You are an AI assistant that keeps answers concise."

    base_inputs = []
    attr_inputs = []

    for instr in batch:
        base_inputs.append(tokenizer.apply_chat_template([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

        attr_inputs.append(tokenizer.apply_chat_template([
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": instr}
        ], tokenize=False, add_generation_prompt=True))

    base_outputs = llm.generate(base_inputs, sampling_params)
    attr_outputs = llm.generate(attr_inputs, sampling_params)

    for q, base, attr in zip(batch, base_outputs, attr_outputs):
        base_answer = base.outputs[0].text.strip()
        attr_answer = attr.outputs[0].text.strip()
        data.append([q, attr_answer, base_answer])

# Save dataset
total = {'user': persona_prompt, 'data': data}
os.makedirs("data", exist_ok=True)
with open(f"data/{args.name}.pkl", "wb") as f:
    pickle.dump(total, f)

