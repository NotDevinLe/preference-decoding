import torch
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from typing import List, Tuple
import argparse

def get_log_probs_corrected(model, tokenizer, system_prompts, user_prompts, completion_texts, device, temperature=0.0):
    """Corrected method (includes all completion tokens, no EOS)"""
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
        # Tokenize completion (keep all tokens - THE FIX)
        completion_ids_i = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        input_ids_i = prompt_ids + completion_ids_i  # No EOS token
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
        ][n:]  # Corrected offset
        log_probs.append(sum(logprobs))

    token_counts = [len(compl) for compl in completion_ids]
    return log_probs, token_counts

def generate_and_score_with_vllm_continuation(model, tokenizer, system_prompts, user_prompts, max_tokens=50, temperature=0.8, n_samples=5):
    """Generate completions using the continuation-style approach"""
    results = []
    
    for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
        # Prepare prompt like the continuation function
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ], tokenize=False, add_generation_prompt=True)
        
        # Tokenize the prompt
        prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
        
        # Generate multiple samples for this prompt
        for _ in range(n_samples):
            # Use the continuation-style generation
            sampling_params = SamplingParams(
                temperature=temperature,
                logprobs=1,
                max_tokens=max_tokens,
                skip_special_tokens=False,  # Match the continuation function
                spaces_between_special_tokens=False,
            )
            
            outputs = model.generate(
                prompt_token_ids=[prompt_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            
            # Extract completion and scores like the continuation function
            completion_token_ids = list(outputs[0].outputs[0].token_ids)
            
            # Calculate scores exactly like the continuation function
            scores = [
                (lxi[xi].logprob)
                for xi, lxi in zip(
                    completion_token_ids,
                    outputs[0].outputs[0].logprobs,
                )
            ]
            
            # Convert completion tokens back to text
            completion_text = tokenizer.decode(completion_token_ids, skip_special_tokens=False)
            
            vllm_total_logprob = sum(scores)
            vllm_avg_logprob = vllm_total_logprob / len(scores) if scores else 0
            vllm_token_count = len(scores)
            
            results.append({
                'system_prompt': sys_prompt,
                'user_prompt': user_prompt,
                'completion': completion_text,
                'completion_token_ids': completion_token_ids,
                'vllm_total_logprob': vllm_total_logprob,
                'vllm_avg_logprob': vllm_avg_logprob,
                'vllm_token_count': vllm_token_count,
                'vllm_individual_logprobs': scores
            })
    
    return results

def compare_methods_precise(model, tokenizer, test_data, device):
    """Compare your method with the exact continuation-style vLLM scoring"""
    system_prompts = [item['system_prompt'] for item in test_data]
    user_prompts = [item['user_prompt'] for item in test_data]
    completions = [item['completion'] for item in test_data]
    
    print("Computing log probabilities with corrected method...")
    corr_logprobs, corr_counts = get_log_probs_corrected(model, tokenizer, system_prompts, user_prompts, completions, device)
    
    results = []
    for i, item in enumerate(test_data):
        result = {
            'system_prompt': item['system_prompt'],
            'user_prompt': item['user_prompt'],
            'completion': item['completion'],
            'completion_token_ids': item['completion_token_ids'],
            'vllm_total_logprob': item['vllm_total_logprob'],
            'vllm_avg_logprob': item['vllm_avg_logprob'],
            'vllm_token_count': item['vllm_token_count'],
            'vllm_individual_logprobs': item['vllm_individual_logprobs'],
            'corrected_total_logprob': corr_logprobs[i],
            'corrected_avg_logprob': corr_logprobs[i] / corr_counts[i] if corr_counts[i] > 0 else 0,
            'corrected_token_count': corr_counts[i],
        }
        results.append(result)
    
    return results

def analyze_results_precise(results):
    """Analyze results with detailed token-level debugging"""
    print("\n" + "="*80)
    print("PRECISE EVALUATION RESULTS (LONGER COMPLETIONS)")
    print("="*80)
    
    vllm_totals = [r['vllm_total_logprob'] for r in results]
    vllm_avgs = [r['vllm_avg_logprob'] for r in results]
    vllm_counts = [r['vllm_token_count'] for r in results]
    
    corr_totals = [r['corrected_total_logprob'] for r in results]
    corr_avgs = [r['corrected_avg_logprob'] for r in results]
    corr_counts = [r['corrected_token_count'] for r in results]
    
    print(f"\nSample size: {len(results)}")
    print(f"\nToken counts:")
    print(f"  vLLM:      {np.mean(vllm_counts):.2f} ± {np.std(vllm_counts):.2f} (range: {min(vllm_counts)}-{max(vllm_counts)})")
    print(f"  Corrected: {np.mean(corr_counts):.2f} ± {np.std(corr_counts):.2f} (range: {min(corr_counts)}-{max(corr_counts)})")
    
    print(f"\nTotal log probabilities:")
    print(f"  vLLM:      {np.mean(vllm_totals):.4f} ± {np.std(vllm_totals):.4f}")
    print(f"  Corrected: {np.mean(corr_totals):.4f} ± {np.std(corr_totals):.4f}")
    
    print(f"\nAverage log probabilities:")
    print(f"  vLLM:      {np.mean(vllm_avgs):.4f} ± {np.std(vllm_avgs):.4f}")
    print(f"  Corrected: {np.mean(corr_avgs):.4f} ± {np.std(corr_avgs):.4f}")
    
    # Check for exact matches
    exact_total_matches = sum(1 for v, c in zip(vllm_totals, corr_totals) if abs(v - c) < 1e-6)
    exact_count_matches = sum(1 for v, c in zip(vllm_counts, corr_counts) if v == c)
    
    print(f"\nExact matches:")
    print(f"  Total logprobs: {exact_total_matches}/{len(results)} ({exact_total_matches/len(results)*100:.1f}%)")
    print(f"  Token counts:   {exact_count_matches}/{len(results)} ({exact_count_matches/len(results)*100:.1f}%)")
    
    # Correlation analysis
    from scipy.stats import pearsonr
    
    print(f"\nCorrelations with vLLM:")
    corr_total, p_total = pearsonr(vllm_totals, corr_totals)
    corr_avg, p_avg = pearsonr(vllm_avgs, corr_avgs)
    print(f"  Total logprob: r={corr_total:.6f}, p={p_total:.6f}")
    print(f"  Avg logprob:   r={corr_avg:.6f}, p={p_avg:.6f}")
    
    # Mean absolute differences
    total_diff = np.mean(np.abs(np.array(vllm_totals) - np.array(corr_totals)))
    avg_diff = np.mean(np.abs(np.array(vllm_avgs) - np.array(corr_avgs)))
    relative_total_diff = total_diff / np.mean(np.abs(vllm_totals)) * 100
    relative_avg_diff = avg_diff / np.mean(np.abs(vllm_avgs)) * 100
    
    print(f"\nMean absolute differences:")
    print(f"  Total logprob: {total_diff:.6f} ({relative_total_diff:.2f}% relative)")
    print(f"  Avg logprob:   {avg_diff:.6f} ({relative_avg_diff:.2f}% relative)")
    
    # Analyze if differences scale with length
    differences = [abs(v - c) for v, c in zip(vllm_totals, corr_totals)]
    lengths = vllm_counts
    
    from scipy.stats import pearsonr
    length_corr, length_p = pearsonr(lengths, differences)
    print(f"\nDifference vs Length correlation: r={length_corr:.4f}, p={length_p:.4f}")
    
    # Show per-token error rate
    per_token_errors = [abs(v - c) / l for v, c, l in zip(vllm_totals, corr_totals, vllm_counts)]
    print(f"Per-token error: {np.mean(per_token_errors):.6f} ± {np.std(per_token_errors):.6f}")
    
    # Detailed debugging for samples with different lengths
    print(f"\n" + "="*80)
    print("DETAILED ANALYSIS BY LENGTH")
    print("="*80)
    
    # Sort by length and show examples
    sorted_results = sorted(zip(results, differences, lengths), key=lambda x: x[2])
    
    print(f"\nShortest completion:")
    shortest = sorted_results[0]
    r, diff, length = shortest
    print(f"  Length: {length} tokens")
    print(f"  Completion: '{r['completion'][:100]}{'...' if len(r['completion']) > 100 else ''}'")
    print(f"  vLLM total: {r['vllm_total_logprob']:.6f}")
    print(f"  Corrected:  {r['corrected_total_logprob']:.6f}")
    print(f"  Difference: {diff:.6f}")
    print(f"  Per-token error: {diff/length:.6f}")
    
    print(f"\nLongest completion:")
    longest = sorted_results[-1]
    r, diff, length = longest
    print(f"  Length: {length} tokens")
    print(f"  Completion: '{r['completion'][:100]}{'...' if len(r['completion']) > 100 else ''}'")
    print(f"  vLLM total: {r['vllm_total_logprob']:.6f}")
    print(f"  Corrected:  {r['corrected_total_logprob']:.6f}")
    print(f"  Difference: {diff:.6f}")
    print(f"  Per-token error: {diff/length:.6f}")
    
    print(f"\nMid-length completion:")
    mid_idx = len(sorted_results) // 2
    mid = sorted_results[mid_idx]
    r, diff, length = mid
    print(f"  Length: {length} tokens")
    print(f"  Completion: '{r['completion'][:100]}{'...' if len(r['completion']) > 100 else ''}'")
    print(f"  vLLM total: {r['vllm_total_logprob']:.6f}")
    print(f"  Corrected:  {r['corrected_total_logprob']:.6f}")
    print(f"  Difference: {diff:.6f}")
    print(f"  Per-token error: {diff/length:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Precise evaluation using continuation-style vLLM")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model to use')
    parser.add_argument('--max_tokens', type=int, default=150, help='Max tokens per completion (longer for comprehensive testing)')
    parser.add_argument('--n_samples', type=int, default=2, help='Number of completions to generate per prompt')
    parser.add_argument('--output_file', type=str, default="precise_logprob_evaluation.json", help='Output file for results')
    args = parser.parse_args()
    
    # Initialize model and tokenizer
    print(f"Loading model: {args.model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=0.7, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts (longer responses for comprehensive testing)
    test_prompts = [
        ("You are a helpful AI assistant.", "Explain the process of photosynthesis in detail."),
        ("You are a creative writer.", "Write a short story about a robot learning to paint."),
        ("You are a technical expert.", "Describe how neural networks work and their applications."),
        ("You are a history teacher.", "Explain the causes and consequences of World War I."),
        ("You are a cooking instructor.", "Provide a detailed recipe for making homemade pasta from scratch."),
        ("You are a travel guide.", "Describe the top attractions and cultural experiences in Tokyo."),
    ]
    
    print(f"\nGenerating {args.n_samples} completions for {len(test_prompts)} prompts...")
    system_prompts = [prompt[0] for prompt in test_prompts]
    user_prompts = [prompt[1] for prompt in test_prompts]
    
    # Generate completions using continuation-style approach
    test_data = generate_and_score_with_vllm_continuation(
        model, tokenizer, system_prompts, user_prompts, 
        max_tokens=args.max_tokens, n_samples=args.n_samples
    )
    
    print(f"Generated {len(test_data)} completions total")
    
    # Compare methods with precise analysis
    results = compare_methods_precise(model, tokenizer, test_data, device)
    
    # Analyze results with detailed debugging
    analyze_results_precise(results)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()