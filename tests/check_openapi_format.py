#!/usr/bin/env python3
"""
Script to check OpenAI API format for vLLM completions endpoint.
Tests the response format and log probabilities structure.
"""

import asyncio
import aiohttp
import json
import argparse
from typing import Dict, Any
from transformers import AutoTokenizer


VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def build_test_prompt(tokenizer, sys_prompt: str, user_prompt: str, completion: str):
    """Build a chat-templated prompt + completion for testing"""
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + completion
    
    # Calculate token counts
    prompt_ids = tokenizer([prompt_text], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    comp_ids = tokenizer([completion], return_tensors=None, add_special_tokens=False)["input_ids"][0]
    
    return full_text, len(prompt_ids), len(comp_ids)


def analyze_response_format(response_data: Dict[str, Any]):
    """Analyze and pretty-print the response format"""
    print("=" * 60)
    print("RESPONSE FORMAT ANALYSIS")
    print("=" * 60)
    
    # Top-level structure
    print(f"Top-level keys: {list(response_data.keys())}")
    print(f"Model: {response_data.get('model', 'N/A')}")
    print(f"Object: {response_data.get('object', 'N/A')}")
    print(f"Created: {response_data.get('created', 'N/A')}")
    
    # Usage stats
    if 'usage' in response_data:
        usage = response_data['usage']
        print(f"Usage: {usage}")
    
    # Choices array
    choices = response_data.get('choices', [])
    print(f"Number of choices: {len(choices)}")
    
    if choices:
        choice = choices[0]
        print(f"\nChoice 0 structure:")
        print(f"  Keys: {list(choice.keys())}")
        print(f"  Index: {choice.get('index', 'N/A')}")
        print(f"  Finish reason: {choice.get('finish_reason', 'N/A')}")
        
        # Text field
        text = choice.get('text', '')
        print(f"  Text length: {len(text)} chars")
        print(f"  Text preview: {repr(text[:100])}")
        
        return choice
    
    return None


def analyze_logprobs_format(choice: Dict[str, Any], n_prefix: int, comp_len: int):
    """Analyze log probabilities format in detail"""
    print("\n" + "=" * 60)
    print("LOG PROBABILITIES ANALYSIS")
    print("=" * 60)
    
    if 'logprobs' not in choice:
        print("L No logprobs field found!")
        return
    
    logprobs = choice['logprobs']
    print(f"Logprobs keys: {list(logprobs.keys())}")
    
    # Token logprobs
    if 'token_logprobs' in logprobs:
        token_logprobs = logprobs['token_logprobs']
        print(f"\nToken logprobs:")
        print(f"  Type: {type(token_logprobs)}")
        print(f"  Length: {len(token_logprobs)}")
        print(f"  Expected prompt tokens: {n_prefix}")
        print(f"  Expected completion tokens: {comp_len}")
        print(f"  Expected total: {n_prefix + comp_len}")
        
        # Check for None values
        none_count = sum(1 for x in token_logprobs if x is None)
        print(f"  None values: {none_count}")
        
        # Show first few values
        print(f"  First 5 values: {token_logprobs[:5]}")
        print(f"  Last 5 values: {token_logprobs[-5:]}")
        
        # Analyze completion segment
        completion_logprobs = token_logprobs[n_prefix:n_prefix + comp_len]
        completion_valid = [x for x in completion_logprobs if x is not None]
        print(f"\nCompletion segment analysis:")
        print(f"  Completion logprobs length: {len(completion_logprobs)}")
        print(f"  Valid completion logprobs: {len(completion_valid)}")
        print(f"  Completion sum: {sum(completion_valid)}")
        print(f"  Completion values: {completion_logprobs}")
    
    # Tokens
    if 'tokens' in logprobs:
        tokens = logprobs['tokens']
        print(f"\nTokens:")
        print(f"  Type: {type(tokens)}")
        print(f"  Length: {len(tokens)}")
        print(f"  First 5: {tokens[:5]}")
        print(f"  Last 5: {tokens[-5:]}")
        
        # Show completion tokens
        completion_tokens = tokens[n_prefix:n_prefix + comp_len]
        print(f"  Completion tokens: {completion_tokens}")
    
    # Text offsets
    if 'text_offset' in logprobs:
        text_offset = logprobs['text_offset']
        print(f"\nText offsets:")
        print(f"  Type: {type(text_offset)}")
        print(f"  Length: {len(text_offset)}")
        print(f"  First 5: {text_offset[:5]}")
        print(f"  Last 5: {text_offset[-5:]}")
    
    # Top logprobs (if available)
    if 'top_logprobs' in logprobs:
        top_logprobs = logprobs['top_logprobs']
        print(f"\nTop logprobs:")
        print(f"  Type: {type(top_logprobs)}")
        print(f"  Length: {len(top_logprobs)}")
        if top_logprobs:
            print(f"  First entry: {top_logprobs[0]}")


async def test_api_format(vllm_url: str, model_id: str):
    """Test the API format with a sample request"""
    print("Testing vLLM OpenAI API format...")
    print(f"URL: {vllm_url}")
    print(f"Model: {model_id}")
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test cases
    test_cases = [
        {
            "system": "You are a helpful assistant.",
            "user": "What is the capital of France?",
            "completion": "The capital of France is Paris."
        },
        {
            "system": "You are a grumpy pirate.",
            "user": "Tell me about treasure.",
            "completion": "Arrr! Treasure be the most precious thing!"
        },
        {
            "system": "You are a math tutor.",
            "user": "What is 2+2?",
            "completion": "2+2 equals 4."
        }
    ]
    
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, test_case in enumerate(test_cases):
            print(f"\n{'=' * 80}")
            print(f"TEST CASE {i+1}")
            print(f"{'=' * 80}")
            
            # Build prompt
            full_text, n_prefix, comp_len = build_test_prompt(
                tokenizer, test_case["system"], test_case["user"], test_case["completion"]
            )
            
            print(f"System prompt: {test_case['system']}")
            print(f"User prompt: {test_case['user']}")
            print(f"Completion: {test_case['completion']}")
            print(f"Prefix tokens: {n_prefix}")
            print(f"Completion tokens: {comp_len}")
            print(f"Full text length: {len(full_text)} chars")
            
            # API request
            payload = {
                "model": model_id,
                "prompt": full_text,
                "echo": True,
                "logprobs": 1,  # Request top 1 logprob alternatives
                "max_tokens": 0,  # No generation
                "temperature": 0.0,
                "stream": False
            }
            
            print(f"\nPayload: {json.dumps(payload, indent=2)}")
            
            try:
                async with session.post(vllm_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Analyze response format
                    choice = analyze_response_format(data)
                    
                    # Analyze logprobs if available
                    if choice:
                        analyze_logprobs_format(choice, n_prefix, comp_len)
                    
                    # Save raw response for inspection
                    filename = f"response_case_{i+1}.json"
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"\nRaw response saved to: {filename}")
                    
            except Exception as e:
                print(f"L Error in test case {i+1}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Check vLLM OpenAI API format")
    parser.add_argument("--vllm-url", type=str, default=VLLM_URL, 
                       help="vLLM server URL")
    parser.add_argument("--model-id", type=str, default=MODEL_ID,
                       help="Model ID to test with")
    
    args = parser.parse_args()
    
    asyncio.run(test_api_format(args.vllm_url, args.model_id))


if __name__ == "__main__":
    main()