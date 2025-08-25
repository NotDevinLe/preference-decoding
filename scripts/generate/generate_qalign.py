#!/usr/bin/env python3
"""
Generate QAlign outputs - just generation, no evaluation.
"""

import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_prompts(input_path: str) -> List[Dict]:
    """Load prompts from file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and 'prompt' in data[0]:
            return data
        return [{"prompt": p} for p in data]
    elif isinstance(data, dict) and 'prompts' in data:
        return [{"prompt": p} for p in data['prompts']]
    else:
        raise ValueError(f"Unsupported data format in {input_path}")


def generate_qalign_outputs(
    prompts: List[Dict], 
    model_path: str, 
    num_samples: int = 32, 
    temperature: float = 0.7
) -> List[Dict]:
    """Generate QAlign outputs."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for prompt_data in tqdm(prompts, desc="Generating QAlign outputs"):
        prompt = prompt_data['prompt']
        persona = prompt_data.get('persona', 'A helpful assistant')
        
        # Apply chat template
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(device)
        
        # Generate multiple samples
        outputs = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode only the new tokens
                generated_text = tokenizer.decode(
                    output[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                outputs.append(generated_text)
        
        results.append({
            "prompt": prompt,
            "persona": persona,
            "outputs": outputs,
            "method": "QAlign",
            "num_samples": num_samples,
            "temperature": temperature
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate QAlign outputs")
    
    parser.add_argument("--input", type=str, required=True, help="Input prompts file")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_prompts", type=int, default=None, help="Max prompts to process")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.input)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Generate outputs
    results = generate_qalign_outputs(prompts, args.model, args.num_samples, args.temperature)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} QAlign samples to {args.output}")


if __name__ == "__main__":
    main()