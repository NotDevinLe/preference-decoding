import torch
import torch.nn.functional as F
import os
import pandas as pd

os.environ["TRANSFORMERS_CACHE"] = "/gscratch/ark/devinl6/hf_cache"
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from huggingface_hub import login
import random
from typing import Optional
import argparse

parser = argparse.ArgumentParser(description="My parameterized script")
parser.add_argument("--name", type=str, required=True, help="User id")

args = parser.parse_args()

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

big_model_id   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_bs = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(big_model_id)

"""

We want to sample the random attributes and use models to output responses from them and then also use the base model to output the normal response and form a dataset.

"""

base_prompt = "You are an AI assistant that keeps answers concise."

data = []
samples = 100

model_bs.eval()
device = torch.device("cuda")
df = pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")
df = df.sample(n=samples)

# Grab a random persona
from datasets import load_dataset
users_ds = load_dataset("kaist-ai/Multifaceted-Collection")

train_split = users_ds["train"]
attribute_prompt = train_split[random.randint(0, len(train_split) - 1)]["system"]

for _, row in df.iterrows():
    data.append([row['Question']])

for j in range(samples):
    print(f"Datapoint: {j}")
    question = data[j][0]

    base_message = [
        {"role": "system", "content": base_prompt},
        {"role": "user", "content": question}
    ]

    attribute_message = [
        {"role": "system", "content": attribute_prompt},
        {"role": "user", "content": question}
    ]

    base_model_prompt = tokenizer.apply_chat_template(base_message, tokenize=False, add_generation_prompt=True)
    attribute_model_prompt = tokenizer.apply_chat_template(attribute_message, tokenize=False, add_generation_prompt=True)

    base_inputs = tokenizer(base_model_prompt, return_tensors="pt").to(model_bs.device)
    attribute_inputs = tokenizer(attribute_model_prompt, return_tensors="pt").to(model_bs.device)

    with torch.no_grad():
        base_output = model_bs.generate(
            **base_inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

        attribute_output = model_bs.generate(
            **attribute_inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    base_answer = tokenizer.decode(
        base_output[0][base_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    attribute_answer = tokenizer.decode(
        attribute_output[0][attribute_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    data[j].append(attribute_answer)
    data[j].append(base_answer)

with open(f"data/{args.name}.pkl", 'wb') as f:
    pickle.dump(data, f)
