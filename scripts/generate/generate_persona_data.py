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


def save_checkpoint(
    output_dir: Path,
    persona_idx: int,
    persona_prompt: str,
    responses: List[Dict[str, Any]]
):
    """Save checkpoint for a single persona."""
    persona_dir = output_dir / f"persona_{persona_idx:03d}"
    persona_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "persona_idx": persona_idx,
        "persona_prompt": persona_prompt,
        "num_responses": len(responses),
        "responses": responses
    }
    
    with open(persona_dir / "responses.json", "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir: Path) -> set:
    """Load completed persona indices from checkpoints."""
    completed = set()
    
    if not output_dir.exists():
        return completed
    
    for persona_dir in output_dir.iterdir():
        if persona_dir.is_dir() and persona_dir.name.startswith("persona_"):
            checkpoint_file = persona_dir / "responses.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    if data.get("num_responses", 0) > 0:
                        completed.add(data["persona_idx"])
    
    return completed


def main():
    parser = argparse.ArgumentParser(description="Generate persona response data")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/preference/user1_train.json",
        help="Path to training data file with questions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/persona_responses",
        help="Directory to save persona responses"
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load questions
    questions = load_questions(args.data_file, args.num_questions)
    
    # Load completed personas if resuming
    completed_personas = load_checkpoint(output_dir) if args.resume else set()
    
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
        
        # Save checkpoint
        save_checkpoint(output_dir, persona_idx, persona_prompt, responses)
        print(f"[Persona {persona_idx}] Saved {len(responses)} responses")
    
    # Create summary file
    summary = {
        "num_personas": num_personas,
        "num_questions": len(questions),
        "total_responses": num_personas * len(questions),
        "model": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "completed_personas": list(range(num_personas))
    }
    
    with open(output_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generation complete! Saved to {output_dir}")
    print(f"  - {num_personas} personas")
    print(f"  - {len(questions)} questions each")
    print(f"  - {num_personas * len(questions)} total responses")


if __name__ == "__main__":
    main()