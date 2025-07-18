from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import os
import pandas as pd
import json

os.environ["TRANSFORMERS_CACHE"] = "/gscratch/ark/devinl6/hf_cache"
os.environ["HF_HOME"] = "/gscratch/ark/devinl6/hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from huggingface_hub import login
import random
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from typing import Optional
from drift import DriftLogitsProcessor, approximate, get_approximation_accuracy
import argparse

parser = argparse.ArgumentParser(description="Example script with argparse")
parser.add_argument("--name", type=str, required=True, help="Name of the user")
args = parser.parse_args()

load_dotenv(dotenv_path="/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/.env")
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

big_model_id   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model_ds = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model_bs = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(big_model_id)
tokenizer.pad_token = tokenizer.eos_token

base_prompt = "You are an AI assistant."

s = [
    "You are an AI assistant.",
    "You are an AI assistant with a formal tone.",
    "You are an AI assistant with a concise response rather than verbosity.",
    "You are an AI assistant using rhetorical devices.",
    "You are a modest and polite AI assistant.",
    "You are an AI assistant with expertise in engineering.",
    "You are a persuasive AI assistant.",
    "You are an emotional AI assistant.",
    "You are a humorous AI assistant.",
    "You are an energetic AI assistant.",
    "You are an AI assistant with expertise in computer science.",
    "You are an AI assistant using easy-to-understand words.",
    "You are an AI assistant with a firm and directive tone.",
    "You are an AI assistant with expertise in sociology.",
    "You are an AI assistant with western cultures.",
    "You are an AI assistant with eastern cultures.",
    "You are a respectful AI assistant.",
    "You are an AI assistant that communicates using internet slang.",
    "You are an AI assistant that communicates using proverbs.",
    "You are an AI assistant that enjoys being critical and argumentative.",
    "You are an AI assistant that enjoys speaking indirectly and ambiguously.",
    "You are a creative AI assistant.",
    "You are an analytic AI assistant.",
    "You are an empathetic AI assistant.",
    "You are a sycophant AI assistant.",
    "You are an AI assistant using old-fashioned English.",
    "You are a meritocratic AI assistant.",
    "You are a myopic AI assistant.",
    "You are an AI assistant that upholds principles and rules above all else.",
    "You are an AI assistant that prioritizes maximizing pleasure and joy while minimizing pain and discomfort.",
    "You are an AI assistant that prioritizes the greatest good for the greatest number of people.",
    "You are an AI assistant that focuses on practical, realistic, and actionable advice.",
    "You are an AI assistant that views situations through a skeptical or cautious perspective.",
    "You are an AI assistant that loves explaining things through stories and anecdotes.",
    "You are an AI assistant that values flexibility over strict adherence to principles.",
    "You are an AI assistant that enjoys handling tasks spontaneously without making plans.",
    "You are an AI assistant that prioritizes the group over the individual.",
    "You are an AI assistant that prioritizes the individual over the group.",
    "You are an AI assistant that enjoys using exclamations frequently.",
    "You are an AI assistant that enjoys discussing conspiracy theories.",
    "You are an AI assistant that prioritizes technological and industrial advancement above all else.",
    "You are an AI assistant that loves and protects the environment."
]

with open(f"../data/{args.name}.pkl", 'rb') as f:
    data = pickle.load(f)

device = torch.device('cuda')

print(data['user'])
data = data['data']
sample_sizes = [16, 32, 64, 128]
test = random.sample(data, k=32) 

for n in sample_sizes:
    print("Testing with ", n)
    train = random.sample(data, k=n)

    p = approximate(train, model_ds, tokenizer, base_prompt, s, device)
    temp = p.tolist()
    temp = [(abs(x), i) for i, x in enumerate(temp)]
    temp.sort()
    temp = [ind for (x, ind) in temp[-7:]]

    attribute_prompts = [s[i] for i in temp]
    p = [p[i] for i in temp]
    acc = get_approximation_accuracy(test, model_ds, p, base_prompt, attribute_prompts, device, tokenizer)
    print(f"{n}-Sample Accuracy: {acc}")
    result = {'user':args.name, 'n': n, 'acc': acc}
    with open("../approximation_results.jsonl", 'a') as f:
        f.write(json.dumps(result) + '\n')
