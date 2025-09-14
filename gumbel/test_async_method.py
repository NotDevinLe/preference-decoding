#!/usr/bin/env python3
"""
Test the exact async method from collector_server.py
"""

import asyncio
import time
import argparse
import json
from typing import List
from openai import OpenAI
import torch

# Import your actual data sampler
try:
    from sampler import DataSampler
except ImportError:
    print("Warning: Could not import DataSampler, using mock data")
    DataSampler = None

# Global variables (like in collector_server.py)
vllm_client = None
model_name = None

async def get_log_probs_from_server(system_prompts: List[str], user_prompts: List[str], completion_texts: List[str], temperature: float = 0.0) -> tuple[List[float], List[int]]:
    """Exact copy of the method from collector_server.py"""
    import asyncio
    
    async def single_request(sys_prompt, user_prompt, completion):
        try:
            messages = [
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
            
            response = vllm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=len(completion.split()) + 10,
                temperature=temperature,
                logprobs=True
            )
            
            if response.choices and response.choices[0].logprobs:
                choice = response.choices[0]
                if choice.logprobs and choice.logprobs.content:
                    # Sum logprobs from completion tokens
                    completion_logprob = sum(
                        token_logprob.logprob 
                        for token_logprob in choice.logprobs.content
                        if token_logprob.logprob is not None
                    )
                    return completion_logprob, len(choice.logprobs.content)
                else:
                    return -1.0, len(completion.split())
            else:
                return -1.0, len(completion.split())
                
        except Exception:
            return -1.0, len(completion.split())
    
    # Create all request tasks and run them concurrently
    tasks = [
        single_request(sys_prompt, user_prompt, completion)
        for sys_prompt, user_prompt, completion in zip(system_prompts, user_prompts, completion_texts)
    ]
    
    # Execute all requests in parallel
    results = await asyncio.gather(*tasks)
    
    # Separate log_probs and token_counts
    log_probs = [result[0] for result in results]
    token_counts = [result[1] for result in results]
    
    return log_probs, token_counts

def load_real_data(dataset_path: str, attribute_prompts_path: str, n_samples: int = 100):
    """Load real data from your dataset and attribute prompts"""
    print(f"📂 Loading real data from {dataset_path}...")
    
    try:
        # Load data sampler
        if DataSampler:
            data_sampler = DataSampler(dataset_path=dataset_path)
            
            # Sample real user data
            user_data = data_sampler(
                users_per_batch=min(n_samples, 50), 
                samples_per_user=1, 
                device=torch.device('cpu')
            )
            
            prompts = user_data['prompts'] 
            outputs = user_data['outputs']
            
            print(f"✅ Loaded {len(prompts)} real prompts and outputs")
            
        else:
            # Fallback mock data
            prompts = ["What is the weather like?"] * n_samples
            outputs = ["The weather is sunny and warm today."] * n_samples
            print(f"⚠️  Using mock data ({n_samples} samples)")
        
        # Load attribute prompts
        with open(attribute_prompts_path, 'r') as f:
            loaded_prompts = json.load(f)
        
        if isinstance(loaded_prompts, list):
            attribute_prompts = loaded_prompts
        elif isinstance(loaded_prompts, dict) and 'prompts' in loaded_prompts:
            attribute_prompts = loaded_prompts['prompts']
        else:
            attribute_prompts = ["You are a helpful assistant."] * 10
            
        print(f"✅ Loaded {len(attribute_prompts)} attribute prompts")
        
        return prompts, outputs, attribute_prompts
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Using fallback mock data...")
        
        # Fallback data
        prompts = ["What is the weather like?"] * n_samples
        outputs = ["The weather is sunny and warm today."] * n_samples
        attribute_prompts = ["You are a helpful assistant.", "You are a creative writer."]
        
        return prompts, outputs, attribute_prompts

async def test_method_with_real_data(prompts, outputs, attribute_prompts, batch_size: int = 4):
    """Test the collector method with real data - mimics actual collector behavior"""
    
    # Calculate total requests like real collector: baseline + all attributes for each batch
    d = len(attribute_prompts)
    total_requests = batch_size * (d + 1)  # +1 for baseline
    
    print(f"🚀 Testing with REAL DATA (like real collector)...")
    print(f"   Batch size: {batch_size}")
    print(f"   Attributes: {d}")
    print(f"   Total concurrent requests: {total_requests} ({batch_size} baseline + {batch_size * d} attribute)")
    
    # Prepare test data exactly like real collector does
    system_prompts = []
    user_prompts = []
    completion_texts = []
    
    base_prompt = "You are a helpful assistant."
    
    # Add baseline requests (one for each sample in batch)
    for i in range(batch_size):
        system_prompts.append(base_prompt)
        user_prompts.append(prompts[i % len(prompts)])
        completion_texts.append(outputs[i % len(outputs)])
    
    # Add ALL attribute requests (each attribute for each sample in batch)
    for attr_idx in range(d):
        attr_prompt = attribute_prompts[attr_idx]
        for i in range(batch_size):
            system_prompts.append(attr_prompt)
            user_prompts.append(prompts[i % len(prompts)])
            completion_texts.append(outputs[i % len(outputs)])
    
    print(f"   System prompts breakdown:")
    print(f"     - Baseline: {batch_size} requests")
    print(f"     - Each of {d} attributes: {batch_size} requests each")
    print(f"   Total: {len(system_prompts)} requests")
    print(f"   Using {len(prompts)} different user prompts/outputs")
    
    # Time the exact method
    start_time = time.time()
    
    try:
        log_probs, token_counts = await get_log_probs_from_server(
            system_prompts, user_prompts, completion_texts, temperature=0.0
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ SUCCESS!")
        print(f"   Total time: {elapsed_time:.2f} seconds")
        print(f"   Time per request: {elapsed_time/total_requests:.3f} seconds")
        print(f"   Requests per second: {total_requests/elapsed_time:.1f}")
        print(f"   Results: {len(log_probs)} log probs, {len(token_counts)} token counts")
        print(f"   Sample log prob: {log_probs[0]:.3f}")
        print(f"   Sample token count: {token_counts[0]}")
        print(f"   Log prob range: [{min(log_probs):.3f}, {max(log_probs):.3f}]")
        
        return True, elapsed_time, batch_size, d
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ FAILED after {elapsed_time:.2f} seconds")
        print(f"   Error: {e}")
        return False, elapsed_time, batch_size, d

async def test_scale(prompts, outputs, attribute_prompts, batch_sizes=[1, 2, 4, 8, 16]):
    """Test performance at different batch sizes (like real collector)"""
    print("\n📈 BATCH SCALING TEST")
    print("=" * 60)
    
    d = len(attribute_prompts)
    results = []
    
    for batch_size in batch_sizes:
        total_requests = batch_size * (d + 1)
        print(f"\n🧪 Testing batch_size={batch_size} ({total_requests} total requests)...")
        success, elapsed_time, bs, attrs = await test_method_with_real_data(prompts, outputs, attribute_prompts, batch_size)
        
        if success:
            rate = total_requests / elapsed_time
            results.append((batch_size, total_requests, elapsed_time, rate))
            print(f"   ✅ Batch {batch_size}: {elapsed_time:.2f}s ({rate:.1f} req/s)")
        else:
            results.append((batch_size, total_requests, None, None))
            print(f"   ❌ Batch {batch_size}: FAILED")
    
    print(f"\n📊 BATCH SCALING RESULTS")
    print("=" * 70)
    print("Batch | Total | Time  | Rate     | Time/Batch | Efficiency")
    print("Size  | Reqs  | (s)   | (req/s)  | (s)        | (%)")
    print("-" * 70)
    
    for i, (batch_size, total_requests, elapsed_time, rate) in enumerate(results):
        if elapsed_time:
            time_per_batch = elapsed_time  # This is time to process one batch
            # Efficiency relative to smallest batch
            if results[0][3]:  # First result has valid rate
                expected_rate_per_batch = results[0][3] / results[0][0]  # rate per sample
                actual_rate_per_batch = rate / batch_size
                efficiency = (actual_rate_per_batch / expected_rate_per_batch) * 100 if expected_rate_per_batch > 0 else 0
            else:
                efficiency = 0
            
            print(f"{batch_size:4d}  | {total_requests:4d}  | {elapsed_time:4.1f}  | {rate:7.1f}  | {time_per_batch:8.2f}   | {efficiency:6.1f}")
        else:
            print(f"{batch_size:4d}  | {total_requests:4d}  | FAIL  | FAIL     | FAIL       | FAIL")

async def main():
    global vllm_client, model_name
    
    parser = argparse.ArgumentParser(description="Test collector's async log prob method with real data")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000/v1", help="VLLM server URL")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (like collector)")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset (optional)")
    parser.add_argument("--attribute-prompts-path", type=str, help="Path to attribute prompts (optional)")
    parser.add_argument("--scale-test", action="store_true", help="Run scaling test with multiple sizes")
    
    args = parser.parse_args()
    
    print("🧪 Testing Collector's Async Method with REAL DATA")
    print("=" * 60)
    print(f"Server: {args.server_url}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Initialize client (like in collector)
    vllm_client = OpenAI(base_url=args.server_url, api_key="dummy")
    model_name = args.model_name
    
    # Test connectivity first
    print("🔗 Testing server connectivity...")
    try:
        models = vllm_client.models.list()
        available_models = [m.id for m in models.data]
        print(f"✅ Connected! Available models: {available_models}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure VLLM server is running!")
        return
    
    print()
    
    # Load real data
    if args.dataset_path and args.attribute_prompts_path:
        prompts, outputs, attribute_prompts = load_real_data(
            args.dataset_path, 
            args.attribute_prompts_path, 
            n_samples=100
        )
    else:
        print("⚠️  No dataset/attribute prompts provided, using fallback data")
        print("   Use --dataset-path and --attribute-prompts-path for real data")
        prompts = ["What is the weather like today?"] * 50
        outputs = ["The weather is sunny and warm with clear skies."] * 50  
        attribute_prompts = [
            "You are a helpful assistant.",
            "You are a creative writer.",
            "You are a technical expert.",
            "You are a friendly teacher.",
            "You are a thoughtful advisor."
        ]
    
    print()
    
    if args.scale_test:
        # Run scaling test
        await test_scale(prompts, outputs, attribute_prompts, [1, 2, 4, 8, 16, 32])
    else:
        # Run single test
        success, elapsed_time, batch_size, d = await test_method_with_real_data(prompts, outputs, attribute_prompts, args.batch_size)
        
        total_requests = batch_size * (d + 1)
        
        print()
        print("🏁 SUMMARY")
        print("=" * 50)
        
        if success:
            print(f"✅ Method works correctly with real data")
            print(f"📊 Batch size: {batch_size}, Attributes: {d}")
            print(f"⏱️  Performance: {elapsed_time:.2f}s for {total_requests} total requests")
            print(f"🚀 Rate: {total_requests/elapsed_time:.1f} requests/second")
            
            # Performance analysis
            time_per_req = elapsed_time / total_requests
            time_per_batch = elapsed_time
            
            print(f"⏱️  Time per request: {time_per_req:.3f}s")
            print(f"⏱️  Time per batch: {time_per_batch:.2f}s")
            
            if time_per_req > 1.0:
                print(f"⚠️  VERY SLOW: {time_per_req:.2f}s per request")
                print("   This is much too slow for production use")
            elif time_per_req > 0.5:
                print(f"🐌 SLOW: {time_per_req:.2f}s per request") 
                print("   Consider optimizing VLLM server or reducing batch size")
            elif time_per_req > 0.1:
                print(f"🐢 MODERATE: {time_per_req:.2f}s per request")
                print("   Acceptable but could be better")
            else:
                print(f"🚀 FAST: {time_per_req:.2f}s per request")
                print("   Good performance!")
                
        else:
            print("❌ Method failed - check logs above")
    
    print("\n💡 To run scaling test: add --scale-test flag")
    print("💡 To use real data: add --dataset-path and --attribute-prompts-path")

if __name__ == "__main__":
    asyncio.run(main())