#!/usr/bin/env python3
import os
import sys
import json
import requests

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

def call_model(messages, temperature=0.7, max_tokens=1024):
    """Call the vLLM model with OpenAI-compatible API"""
    url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"  # vLLM doesn't need real auth
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling model: {e}")
        return None

def main():
    # Example 1: Simple chat
    print("=== Simple Chat Example ===")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    result = call_model(messages)
    if result:
        content = result["choices"][0]["message"]["content"]
        print(f"Response: {content}\n")
    
    # Example 2: Persona evaluation
    print("=== Persona Evaluation Example ===")
    persona = "A grumpy old wizard who speaks in riddles"
    response_a = "Magic is everywhere, young one. Look closer."
    response_b = "Bah! Magic schmагic. *waves staff irritably* Why must ye always seek what hides in plain sight, hmm?"
    
    eval_messages = [
        {"role": "user", "content": f"""
You are evaluating which response better matches this persona:
PERSONA: {persona}

RESPONSE A: {response_a}
RESPONSE B: {response_b}

Judge which response better captures:
1. Personality traits and speaking style
2. Consistency with the character
3. Authenticity of the persona

Provide your reasoning and choose A or B.
"""}
    ]
    
    result = call_model(eval_messages, temperature=0.1)  # Low temp for consistent evaluation
    if result:
        evaluation = result["choices"][0]["message"]["content"]
        print(f"Evaluation: {evaluation}\n")
    
    # Example 3: Batch processing multiple prompts
    print("=== Batch Processing Example ===")
    prompts = [
        "Write a haiku about programming",
        "Explain quantum computing in one sentence",
        "What's the best pizza topping?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        result = call_model(messages, temperature=0.8, max_tokens=200)
        
        if result:
            content = result["choices"][0]["message"]["content"]
            print(f"Prompt {i}: {prompt}")
            print(f"Response: {content}\n")
    
    # Example 4: Interactive mode
    print("=== Interactive Mode (type 'quit' to exit) ===")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        messages = [{"role": "user", "content": user_input}]
        result = call_model(messages)
        
        if result:
            content = result["choices"][0]["message"]["content"]
            print(f"Assistant: {content}\n")

if __name__ == "__main__":
    main()