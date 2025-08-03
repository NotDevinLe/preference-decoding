from drift import DriftLogitsProcessor
import torch
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
import numpy as np
import pickle
from dotenv import load_dotenv
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from vllm import LLM, SamplingParams
import gc
import cvxpy as cp
import json
from attribute_prompts import attribute_prompts, base_prompt

with open('../results/preference/user1_p.json', 'r') as f:
    p_list = json.load(f)

with open('../data/bon.json', 'r') as f:
    data = json.load(f)

prompts = []
for entry in data:
    prompts.append(entry['prompt'])

big_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(small_model_id)

# Setup quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

big_model = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

small_model = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

# Create drift logits processor
drift_processor = DriftLogitsProcessor(
    b=0.5,  # strength parameter
    small_model=small_model,
    tokenizer=tokenizer,
    base_prompt=base_prompt,
    attribute_prompts=attribute_prompts,
    weights=p_list
)

# Setup generation parameters
logits_processor_list = LogitsProcessorList([drift_processor])

# Generate responses for each prompt
results = []
for i, prompt in enumerate(prompts):
    print(f"Generating response {i+1}/{len(prompts)}")
    
    # Format prompt for generation
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(big_model.device)
    
    # Generate with drift decoding
    with torch.no_grad():
        output = big_model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            logits_processor=logits_processor_list,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    results.append({
        "prompt": prompt,
        "response": response
    })

# Save results
with open('../results/drift_decoding_responses.json', 'w') as f:
    json.dump(results, f, indent=2)