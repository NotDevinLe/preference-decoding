from drift import DriftLogitsProcessor
import torch
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from attribute_prompts import attribute_prompts, base_prompt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--b', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=4)  # Add batch size parameter
args = parser.parse_args()

# Load data
with open('../results/preference/user1_p.json', 'r') as f: 
    p_list = json.load(f)

with open('../data/bon.json', 'r') as f:
    data = json.load(f)

prompts = [entry['prompt'] for entry in data]

big_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load models WITHOUT quantization for speed
print("Loading models (no quantization for speed)...")
big_model = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

small_model = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Set models to eval mode
big_model.eval()
small_model.eval()

# Find p vector
p = None
for entry in p_list:
    if entry['lambda'] == 0.01 and entry['sample_size'] == 200:
        p = entry['p']
        break

if p is None:
    raise ValueError("Could not find p vector with lambda=0.01 and sample_size=200")

print(f"Using p vector: {p[:5]}...")  # Print first 5 elements

# Batch generation function
def generate_batch(prompts_batch, big_model, drift_processor, tokenizer, max_new_tokens=512):
    # Format all prompts
    formatted_prompts = []
    for prompt in prompts_batch:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted_prompt)
    
    # Tokenize batch
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048
    ).to(big_model.device)
    
    # Generate batch
    with torch.no_grad():
        outputs = big_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            logits_processor=[drift_processor],
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs['attention_mask'],
            use_cache=True  # Enable KV caching for speed
        )
    
    # Decode responses
    responses = []
    for i, output in enumerate(outputs):
        # Get only the generated part (skip input tokens)
        generated_tokens = output[inputs['input_ids'][i].shape[0]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)
    
    return responses

# Create drift logits processor
drift_processor = DriftLogitsProcessor(
    b=args.b / 10,
    small_model=small_model,
    tokenizer=tokenizer,
    base_prompt=base_prompt,
    attribute_prompts=attribute_prompts,
    weights=p
)

# Process in batches
results = []
batch_size = args.batch_size

print(f"Processing {len(prompts)} prompts in batches of {batch_size}")

for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
    
    try:
        batch_responses = generate_batch(batch_prompts, big_model, drift_processor, tokenizer)
        
        for prompt, response in zip(batch_prompts, batch_responses):
            results.append({
                "prompt": prompt,
                "response": response
            })
            print(f"Generated response for: {prompt[:50]}...")
            
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM! Reducing batch size and trying again...")
        # Fallback to individual generation for this batch
        for prompt in batch_prompts:
            try:
                batch_responses = generate_batch([prompt], big_model, drift_processor, tokenizer)
                results.append({
                    "prompt": prompt,
                    "response": batch_responses[0]
                })
            except Exception as e:
                print(f"Failed to generate for prompt: {e}")
                results.append({
                    "prompt": prompt,
                    "response": "[GENERATION_FAILED]"
                })
    
    # Clear cache periodically
    if i % (batch_size * 4) == 0:
        torch.cuda.empty_cache()

# Save results
output_file = f'../results/drift_decoding_responses_b{args.b}_batch{batch_size}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved {len(results)} responses to {output_file}")
print(f"Average response length: {np.mean([len(r['response']) for r in results]):.1f} chars")