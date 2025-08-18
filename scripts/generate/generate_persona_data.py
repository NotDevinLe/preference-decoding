#!/usr/bin/env python3
"""
Generate responses for all personas using the same set of questions.
This creates the data needed for sparse coding attribute selection.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from attributes.personas import persona_prompts


def load_questions(data_file: str, num_questions: int = 100) -> List[str]:
    """Load questions from training data file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    questions = [item['prompt'] for item in data[:num_questions]]
    print(f"Loaded {len(questions)} questions from {data_file}")
    return questions


def generate_persona_responses(
    model: LLM,
    tokenizer: AutoTokenizer,
    persona_prompt: str,
    questions: List[str],
    batch_size: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> List[Dict[str, Any]]:
    """Generate responses for a single persona across all questions."""
    
    # Prepare all prompts
    formatted_prompts = []
    for question in questions:
        formatted = tokenizer.apply_chat_template([
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question}
        ], tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    
    # Generate responses in batches
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    all_responses = []
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        outputs = model.generate(batch, sampling_params)
        
        for question, output in zip(batch_questions, outputs):
            response_text = output.outputs[0].text
            all_responses.append({
                "question": question,
                "response": response_text
            })
    
    return all_responses


def save_all_data(
    output_file: Path,
    all_data: Dict[str, Any]
):
    """Save all persona data to a single file."""
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved all data to {output_file}")


def load_checkpoint(output_file: Path) -> Dict[str, Any]:
    """Load existing data from checkpoint file."""
    if output_file.exists():
        with open(output_file, 'r') as f:
            return json.load(f)
    return {
        "metadata": {},
        "personas": [],
        "completed_indices": []
    }


def main():
    parser = argparse.ArgumentParser(description="Generate persona response data")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/preference/user1_train.json",
        help="Path to training data file with questions"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/persona_responses.json",
        help="Output file to save all persona responses"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to use"
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        default=None,
        help="Number of personas to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    questions = load_questions(args.data_file, args.num_questions)
    
    # Load existing data if resuming
    all_data = load_checkpoint(output_file) if args.resume else {
        "metadata": {},
        "personas": [],
        "completed_indices": []
    }
    
    completed_personas = set(all_data.get("completed_indices", []))
    
    # Determine personas to process
    num_personas = args.num_personas or len(persona_prompts)
    num_personas = min(num_personas, len(persona_prompts))
    
    print(f"\nGenerating responses for {num_personas} personas")
    print(f"Using {len(questions)} questions")
    print(f"Total generations: {num_personas * len(questions)}")
    
    if completed_personas:
        print(f"Resuming from checkpoint - {len(completed_personas)} personas already completed")
    
    # Initialize model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        dtype="bfloat16" if args.device == "cuda" else "float32"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Update metadata
    all_data["metadata"] = {
        "num_personas": num_personas,
        "num_questions": len(questions),
        "total_responses": num_personas * len(questions),
        "model": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "questions": questions
    }
    
    # Generate responses for each persona
    for persona_idx in tqdm(range(num_personas), desc="Processing personas"):
        if persona_idx in completed_personas:
            print(f"Skipping persona {persona_idx} (already completed)")
            continue
        
        persona_prompt = persona_prompts[persona_idx]
        print(f"\n[Persona {persona_idx}] Generating responses...")
        
        # Generate responses
        responses = generate_persona_responses(
            model=model,
            tokenizer=tokenizer,
            persona_prompt=persona_prompt,
            questions=questions,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Add to all_data
        persona_data = {
            "persona_idx": persona_idx,
            "persona_prompt": persona_prompt,
            "responses": responses
        }
        
        # Check if this persona already exists in the data
        existing_idx = None
        for i, p in enumerate(all_data["personas"]):
            if p["persona_idx"] == persona_idx:
                existing_idx = i
                break
        
        if existing_idx is not None:
            all_data["personas"][existing_idx] = persona_data
        else:
            all_data["personas"].append(persona_data)
        
        # Update completed indices
        if persona_idx not in all_data["completed_indices"]:
            all_data["completed_indices"].append(persona_idx)
        
        # Save checkpoint after each persona
        save_all_data(output_file, all_data)
        print(f"[Persona {persona_idx}] Saved {len(responses)} responses")
    
    # Final save
    save_all_data(output_file, all_data)
    
    print(f"\n✓ Generation complete! Saved to {output_file}")
    print(f"  - {num_personas} personas")
    print(f"  - {len(questions)} questions each")
    print(f"  - {num_personas * len(questions)} total responses")
    
    # Print file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  - File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()