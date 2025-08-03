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

# Setup output file
output_file = f'../results/drift_decoding_responses_b{args.b}_batch{args.batch_size}.json'

# Initialize or load existing results
results = []
start_idx = 0

# Check if output file exists and load existing results
if os.path.exists(output_file):
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"📁 Resuming from {start_idx} existing results")
    except:
        print("⚠️ Could not load existing results, starting fresh")
        results = []

def save_results_incrementally(results, output_file):
    """Save results to file immediately"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# Process in batches
batch_size = args.batch_size
total_batches = (len(prompts) - start_idx - 1) // batch_size + 1

print(f"Processing {len(prompts) - start_idx} remaining prompts in batches of {batch_size}")

for i in range(start_idx, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_num = (i - start_idx) // batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches} (prompts {i+1}-{min(i+batch_size, len(prompts))}/{len(prompts)})")
    
    batch_results = []
    
    try:
        batch_responses = generate_batch(batch_prompts, big_model, drift_processor, tokenizer)
        
        for prompt, response in zip(batch_prompts, batch_responses):
            batch_results.append({
                "prompt": prompt,
                "response": response
            })
            print(f"✅ Generated response for: {prompt[:50]}...")
            
    except torch.cuda.OutOfMemoryError:
        print("💥 CUDA OOM! Reducing batch size and trying again...")
        # Fallback to individual generation for this batch
        for prompt in batch_prompts:
            try:
                batch_responses = generate_batch([prompt], big_model, drift_processor, tokenizer)
                batch_results.append({
                    "prompt": prompt,
                    "response": batch_responses[0]
                })
                print(f"✅ Generated response (single): {prompt[:50]}...")
            except Exception as e:
                print(f"❌ Failed to generate for prompt: {e}")
                batch_results.append({
                    "prompt": prompt,
                    "response": "[GENERATION_FAILED]"
                })
    
    # Add batch results to main results and save immediately
    results.extend(batch_results)
    save_results_incrementally(results, output_file)
    print(f"💾 Saved batch {batch_num} - Total: {len(results)} responses")
    
    # Clear cache periodically
    if i % (batch_size * 4) == 0:
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache")

print(f"🎉 Completed! Saved {len(results)} total responses to {output_file}")
if results:
    avg_length = np.mean([len(r['response']) for r in results if r['response'] != '[GENERATION_FAILED]'])
    print(f"📊 Average response length: {avg_length:.1f} chars")