import numpy as np
import random
import itertools
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from drift import get_training_matrix, approximate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(system_prompt1, system_prompt2, base_prompt, prob1, prob2, size, dolly_ds):
    base_prompt_inputs = []
    base_prompt_outputs = []

    instructions = [build_prompt(row["instruction"], row["context"]) for row in dolly_ds.shuffle().select(range(size))]

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

    for instruction in instructions[:int(len(instructions) * 0.8)]:
        attr1_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt1},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        attr1_prompt_inputs.append(attr1_prompt_input)

    attr1_prompt_outputs = llm.generate(attr1_prompt_inputs, sampling_params)
    attr1_prompt_outputs = [output.outputs[0].text.strip() for output in attr1_prompt_outputs]

    attr2_prompt_inputs = []
    attr2_prompt_outputs = []

    for instruction in instructions[int(len(instructions) * 0.8):]:
        attr2_prompt_input = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": instruction}
        ], tokenize=False, add_generation_prompt=True)
        attr2_prompt_inputs.append(attr2_prompt_input)

    attr2_prompt_outputs = llm.generate(attr2_prompt_inputs, sampling_params)
    attr2_prompt_outputs = [output.outputs[0].text.strip() for output in attr2_prompt_outputs]

    attribute_prompts_outputs = attr1_prompt_outputs + attr2_prompt_outputs

    all_data = []
    for i in range(len(instructions)):
        all_data.append({
            "prompt": instructions[i],
            "chosen": attribute_prompts_outputs[i],
            "rejected": base_prompt_outputs[i]
        })

    return all_data
    

prompts = [
    "You are a humorous AI assistant.",
    "You are an AI assistant with expertise in sociology.",
    "You are an AI assistant that communicates using internet slang.",
    "You are a persuasive AI assistant.",
    "You are an AI assistant that loves explaining things through stories and anecdotes."
]

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

pairs = list(itertools.combinations(prompts, 2))

train_size, test_size = 200, 1000

for attr1, attr2 in pairs:
    prob1 = random.random()
    prob2 = 1 - prob1

    print(f"Evaluating {attr1} and {attr2} with probability {prob1} and {prob2}")
    test, train = generate_data(attr1, attr2, base_prompt, prob1, prob2, train_size, dolly_ds)
    test_data = generate_data(attr1, attr2, base_prompt, prob1, prob2, test_size, dolly_ds)

    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    print(f"Test data: {test}")
    print(f"Train data: {train}")

    p = approximate(train, llm, tokenizer, base_prompt, prompts, device)
    p = p.cpu().numpy()
    p = np.mean(p, axis=0)
    if np.linalg.norm(p, ord=1) > 1:
        p = p * (1 / np.linalg.norm(p, ord=1))
    
    print(f"P: {p}")
    print(f"The top two attributes are {prompts[np.argmax(p)]} and {prompts[np.argsort(p)[-2]]}")
