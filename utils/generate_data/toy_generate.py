import os
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import random

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--sample_size", type=int, required=True)
parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Generate train or test split")
parser.add_argument("--save_path", type=str, default="../../data/preference", help="Path to save preference data")
args = parser.parse_args()

sample_size = args.sample_size

base_prompt = "You are an AI assistant."

# Model setup
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.7,
    max_model_len=8192
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

def build_prompt(instruction, context):
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction

instructions = [build_prompt(row["instruction"], row["context"]) for row in dolly_ds.shuffle().select(range(sample_size))]

all_data = []

base_prompt_inputs = []
base_prompt_outputs = []
for instruction in instructions:
    base_prompt_input = tokenizer.apply_chat_template([
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": instruction}
    ], tokenize=False, add_generation_prompt=True)
    base_prompt_inputs.append(base_prompt_input)

base_prompt_outputs = llm.generate(base_prompt_inputs, sampling_params)
base_prompt_outputs = [output.outputs[0].text.strip() for output in base_prompt_outputs]


attr1_prompt_inputs = []
attr1_prompt_outputs = []

attr1_prompt = "You are an AI assistant that communicates using internet slang."
for instruction in instructions[:int(len(instructions) * 0.8)]:
    attr1_prompt_input = tokenizer.apply_chat_template([
        {"role": "system", "content": attr1_prompt},
        {"role": "user", "content": instruction}
    ], tokenize=False, add_generation_prompt=True)
    attr1_prompt_inputs.append(attr1_prompt_input)

attr1_prompt_outputs = llm.generate(attr1_prompt_inputs, sampling_params)
attr1_prompt_outputs = [output.outputs[0].text.strip() for output in attr1_prompt_outputs]

attr2_prompt_inputs = []
attr2_prompt_outputs = []

attr2_prompt = "You are a persuasive AI assistant."
for instruction in instructions[int(len(instructions) * 0.8):]:
    attr2_prompt_input = tokenizer.apply_chat_template([
        {"role": "system", "content": attr2_prompt},
        {"role": "user", "content": instruction}
    ], tokenize=False, add_generation_prompt=True)
    attr2_prompt_inputs.append(attr2_prompt_input)

attr2_prompt_outputs = llm.generate(attr2_prompt_inputs, sampling_params)
attr2_prompt_outputs = [output.outputs[0].text.strip() for output in attr2_prompt_outputs]

attribute_prompts_outputs = attr1_prompt_outputs + attr2_prompt_outputs

for i in range(len(instructions)):
    all_data.append({
        "prompt": instructions[i],
        "chosen": attribute_prompts_outputs[i],
        "rejected": base_prompt_outputs[i]
    })

random.shuffle(all_data)
with open(f"{args.save_path}/{args.name}_{args.split}.json", "w") as f:
    json.dump(all_data, f, indent=2)