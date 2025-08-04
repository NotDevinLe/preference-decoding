import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from drift import DriftLogitsProcessor  # Your working version
from attribute_prompts import attribute_prompts, base_prompt
import json

# Load models
print("Loading models...")
big_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
small_model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(small_model_id)
tokenizer.pad_token = tokenizer.eos_token

big_model = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

small_model = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load p vector
with open('../results/preference/user1_p.json', 'r') as f:
    p_list = json.load(f)

p = None
for entry in p_list:
    if entry['lambda'] == 0.01 and entry['sample_size'] == 200:
        p = entry['p']
        break

print(f"Using p vector: {p[:5]}... (length: {len(p)})")

# Test prompts
test_prompts = [
    "Write a funny joke about programming.",
    "Explain quantum physics simply.", 
    "What's the best way to learn Python?",
    "Write a short story about a robot.",
    "Give me advice on time management."
]

def generate_comparison(prompt, b_values=[10.0, 1.0, 0.5]):
    """Generate responses with different b values for comparison"""
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}")
    
    results = {}
    
    for b in b_values:
        print(f"\n--- b = {b} ---")
        
        if b >= 10.0:
            # Large b ≈ minimal drift (conservative)
            print("Generating with minimal drift (b=10, ~baseline)...")
        else:
            print(f"Generating with drift (b={b})...")
        
        # Always use drift processor, but with different b values
        drift_processor = DriftLogitsProcessor(
            b=b,
            small_model=small_model,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            attribute_prompts=attribute_prompts,
            weights=p
        )
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(big_model.device)
        
        # Generate (always with drift processor now)
        with torch.no_grad():
            output = big_model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                logits_processor=[drift_processor],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Response: {response}")
        print(f"Length: {len(response)} chars")
        
        results[f"b_{b}"] = response
    
    return results

# Test all prompts
all_results = {}

for prompt in test_prompts:
    all_results[prompt] = generate_comparison(prompt)

# Summary comparison
print(f"\n{'='*80}")
print("SUMMARY: DRIFT EFFECT ANALYSIS")
print(f"{'='*80}")

for prompt, results in all_results.items():
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    baseline = results.get("b_10.0", "")
    drift_1 = results.get("b_1.0", "")
    drift_2 = results.get("b_0.5", "")
    
    print(f"Conservative (b=10): {baseline[:100]}...")
    print(f"Moderate (b=1.0):    {drift_1[:100]}...")
    print(f"Aggressive (b=0.5):  {drift_2[:100]}...")
    
    # Check if responses are different
    if baseline == drift_1 == drift_2:
        print("⚠️  WARNING: All responses identical!")
    elif len(set([baseline, drift_1, drift_2])) == 3:
        print("✅ All responses different - drift is working!")
    else:
        print("⚠️  Some responses identical")

# Quality assessment prompts
print(f"\n{'='*80}")
print("QUALITATIVE ASSESSMENT")
print(f"{'='*80}")
print("Look for these drift effects:")
print("1. Different word choices/style")
print("2. Different response structure") 
print("3. Different tone/personality")
print("4. Responses should still be coherent and on-topic")
print("5. Higher b values should be more conservative")

# Save results for analysis
with open('../results/drift_generation_test.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Results saved to ../results/drift_generation_test.json")