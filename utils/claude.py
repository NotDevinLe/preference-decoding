#!/usr/bin/env python3
"""
Test if system prompts actually affect log probabilities
Compare direct method vs your get_log_probs function
"""
import vllm
from transformers import AutoTokenizer
from vllm import SamplingParams

def get_log_probs(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    """Your original get_log_probs function"""
    input_ids = []
    ns = []
    completion_ids = []
    for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts):
        # Apply chat template to get prompt tokens
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        ns.append(len(prompt_ids))
        # Tokenize completion without skipping tokens
        completion_ids_i = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        input_ids_i = prompt_ids + completion_ids_i + [tokenizer.eos_token_id]
        input_ids.append(input_ids_i)
        completion_ids.append(completion_ids_i)
    
    sampling_params = SamplingParams(
        prompt_logprobs=0,
        max_tokens=1,
        temperature=temperature,
    )

    outputs = model.generate(
        prompt_token_ids=input_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    log_probs = []
    for compl, out, n in zip(input_ids, outputs, ns):
        logprobs = [
            (lxi[xi].logprob)
            for xi, lxi in zip(
                compl[1:],
                out.prompt_logprobs[1:],
            )
        ][n:]
        log_probs.append(sum(logprobs))

    token_counts = [len(compl) for compl in completion_ids]
    return log_probs, token_counts

def test_system_prompt_effects():
    # Initialize model
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = vllm.LLM(model=model_id, tensor_parallel_size=1, gpu_memory_utilization=0.3, max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Test data
    user_prompt = "What is 2+2?"
    completion = "Ahoy! That be 4, matey!"  # Pirate-style completion
    
    system_prompts = [
        "You are an AI assistant that speaks like a pirate.",
        "You are an AI assistant that speaks like a university professor.",
        "You are an AI assistant that speaks like a robot.",
    ]
    
    results_direct = {}
    results_your_method = {}
    
    print("="*80)
    print("TESTING DIRECT METHOD (from previous test)")
    print("="*80)
    
    for system_prompt in system_prompts:
        print(f"\n{'='*60}")
        print(f"TESTING: {system_prompt}")
        print(f"{'='*60}")
        
        # Create prompt
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        prompt_tokens = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        completion_tokens = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        full_tokens = prompt_tokens + completion_tokens + [tokenizer.eos_token_id]
        
        print(f"Prompt length: {len(prompt_tokens)}")
        print(f"Completion length: {len(completion_tokens)}")
        
        # Get logprobs
        sampling_params = SamplingParams(
            prompt_logprobs=0,  # Your original setting
            max_tokens=1,
            temperature=0.0,
        )
        
        outputs = model.generate(
            prompt_token_ids=[full_tokens],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        
        output = outputs[0]
        
        # Extract completion logprobs
        completion_start = len(prompt_tokens)
        completion_logprobs = []
        
        print(f"Completion tokens and logprobs:")
        for i in range(completion_start, completion_start + len(completion_tokens)):
            if i < len(output.prompt_logprobs) and output.prompt_logprobs[i]:
                actual_token = full_tokens[i]
                actual_text = tokenizer.decode([actual_token])
                
                if actual_token in output.prompt_logprobs[i]:
                    logprob = output.prompt_logprobs[i][actual_token].logprob
                    completion_logprobs.append(logprob)
                    print(f"  {repr(actual_text)}: {logprob:.4f}")
                else:
                    print(f"  {repr(actual_text)}: TOKEN_NOT_FOUND")
        
        total_logprob = sum(completion_logprobs)
        avg_logprob = total_logprob / len(completion_logprobs) if completion_logprobs else 0
        
        print(f"\nSUMMARY:")
        print(f"  Total logprob: {total_logprob:.4f}")
        print(f"  Average per token: {avg_logprob:.4f}")
        print(f"  Tokens processed: {len(completion_logprobs)}/{len(completion_tokens)}")
        
        results_direct[system_prompt] = {
            'total': total_logprob,
            'average': avg_logprob,
            'individual': completion_logprobs
        }
    
    # Now test your get_log_probs method
    print(f"\n{'='*80}")
    print("TESTING YOUR get_log_probs METHOD")
    print(f"{'='*80}")
    
    for system_prompt in system_prompts:
        print(f"\nTesting: {system_prompt}")
        
        # Use your method
        log_probs, token_counts = get_log_probs(
            model, tokenizer, 
            [system_prompt], [user_prompt], [completion], 
            device=None, temperature=0.0
        )
        
        total_logprob = log_probs[0]
        avg_logprob = total_logprob / token_counts[0] if token_counts[0] > 0 else 0
        
        print(f"  Your method - Total: {total_logprob:.4f}, Avg: {avg_logprob:.4f}, Tokens: {token_counts[0]}")
        
        results_your_method[system_prompt] = {
            'total': total_logprob,
            'average': avg_logprob,
            'token_count': token_counts[0]
        }
    
    # Compare both methods
    print(f"\n{'='*80}")
    print("COMPARISON: DIRECT vs YOUR METHOD")
    print(f"{'='*80}")
    
    print(f"{'System Prompt':<30} {'Direct Method':<15} {'Your Method':<15} {'Difference':<12}")
    print("-" * 75)
    
    for prompt in system_prompts:
        direct_total = results_direct[prompt]['total']
        your_total = results_your_method[prompt]['total']
        difference = abs(direct_total - your_total)
        
        print(f"{prompt[:30]:<30} {direct_total:<15.4f} {your_total:<15.4f} {difference:<12.6f}")
    
    # Check if methods are equivalent
    print(f"\n{'='*80}")
    print("METHOD VALIDATION")
    print(f"{'='*80}")
    
    max_difference = 0
    for prompt in system_prompts:
        diff = abs(results_direct[prompt]['total'] - results_your_method[prompt]['total'])
        max_difference = max(max_difference, diff)
    
    if max_difference < 0.001:
        print("✅ SUCCESS: Both methods give identical results (within 0.001 tolerance)")
        print("✅ Your get_log_probs function is working correctly!")
    else:
        print(f"❌ WARNING: Methods differ by up to {max_difference:.6f}")
        print("❌ There may be a bug in your get_log_probs function")
    
    # Show the expected large differences
    print(f"\n{'='*80}")
    print("SYSTEM PROMPT EFFECTIVENESS")
    print(f"{'='*80}")
    
    pirate_result = results_your_method[system_prompts[0]]['total']  # Pirate
    professor_result = results_your_method[system_prompts[1]]['total']  # Professor
    robot_result = results_your_method[system_prompts[2]]['total']  # Robot
    
    print(f"Pirate text evaluated under:")
    print(f"  Pirate prompt:     {pirate_result:.4f}")
    print(f"  Professor prompt:  {professor_result:.4f}")
    print(f"  Robot prompt:      {robot_result:.4f}")
    
    print(f"\nDifferences (should be large):")
    print(f"  Pirate vs Professor: {pirate_result - professor_result:.4f}")
    print(f"  Pirate vs Robot:     {pirate_result - robot_result:.4f}")
    print(f"  Professor vs Robot:  {professor_result - robot_result:.4f}")
    
    if abs(pirate_result - professor_result) > 30:
        print("✅ Large differences detected - system prompts are working!")
    else:
        print("❌ Small differences - system prompts may not be working properly")

if __name__ == "__main__":
    test_system_prompt_effects()