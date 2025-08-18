#!/usr/bin/env python3
"""
Prepare questions for persona generation using the same format as BON generation.
Combines Dolly instruction and context fields like in bon_generate.py.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def build_prompt(instruction: str, context: str) -> str:
    """Combine instruction and context like in bon_generate.py"""
    if context.strip():
        return f"{instruction}\n\n{context}"
    else:
        return instruction


def prepare_questions_from_dolly(
    num_questions: int = 100,
    seed: int = 42,
    output_file: str = "data/questions.json"
) -> List[str]:
    """
    Prepare questions from Dolly dataset using the same format as BON generation.
    
    Args:
        num_questions: Number of questions to prepare
        seed: Random seed for reproducibility
        output_file: Output file to save questions
        
    Returns:
        List of prepared questions
    """
    print("Loading Dolly dataset...")
    dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Select random subset with seed for reproducibility
    selected_rows = dolly_ds.shuffle(seed=seed).select(range(num_questions))
    
    # Prepare questions using the same logic as bon_generate.py
    questions = []
    for row in selected_rows:
        question = build_prompt(row["instruction"], row["context"])
        questions.append(question)
    
    # Save questions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    questions_data = {
        "metadata": {
            "num_questions": len(questions),
            "source": "databricks/databricks-dolly-15k",
            "seed": seed,
            "format": "instruction + context (same as bon_generate.py)"
        },
        "questions": questions
    }
    
    with open(output_path, 'w') as f:
        json.dump(questions_data, f, indent=2)
    
    print(f"✅ Saved {len(questions)} questions to {output_file}")
    return questions


def prepare_questions_from_existing(
    input_file: str,
    output_file: str = "data/questions.json",
    max_questions: int = None
) -> List[str]:
    """
    Extract questions from existing training data file.
    
    Args:
        input_file: Path to existing training data (JSON format)
        output_file: Output file to save questions
        max_questions: Maximum number of questions to extract
        
    Returns:
        List of prepared questions
    """
    print(f"Loading questions from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract prompts from existing data
    if isinstance(data, list):
        # List of items with 'prompt' field
        questions = [item['prompt'] for item in data if 'prompt' in item]
    elif isinstance(data, dict) and 'questions' in data:
        # Already in our format
        questions = data['questions']
    else:
        raise ValueError(f"Unsupported data format in {input_file}")
    
    # Limit questions if requested
    if max_questions:
        questions = questions[:max_questions]
    
    # Save questions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    questions_data = {
        "metadata": {
            "num_questions": len(questions),
            "source": input_file,
            "format": "extracted from existing data"
        },
        "questions": questions
    }
    
    with open(output_path, 'w') as f:
        json.dump(questions_data, f, indent=2)
    
    print(f"✅ Saved {len(questions)} questions to {output_file}")
    return questions


def create_preference_format(
    questions: List[str],
    output_file: str = "data/preference_questions.json"
):
    """
    Convert questions to preference dataset format (for compatibility with existing scripts).
    
    Args:
        questions: List of questions
        output_file: Output file path
    """
    preference_data = []
    
    for i, question in enumerate(questions):
        preference_data.append({
            "prompt": question,
            "chosen": "",  # Placeholder - will be filled by persona generation
            "rejected": "",  # Placeholder - will be filled by persona generation
            "id": i
        })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(preference_data, f, indent=2)
    
    print(f"✅ Created preference format with {len(preference_data)} items in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare questions for persona generation")
    parser.add_argument(
        "--source",
        type=str,
        choices=["dolly", "existing"],
        default="dolly",
        help="Source of questions: 'dolly' for Dolly dataset, 'existing' for existing file"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file path (required if source='existing')"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/questions.json",
        help="Output file for questions"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to prepare"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--create-preference-format",
        action="store_true",
        help="Also create preference dataset format file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PREPARING QUESTIONS FOR PERSONA GENERATION")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Output file: {args.output_file}")
    print(f"Number of questions: {args.num_questions}")
    print("")
    
    # Prepare questions based on source
    if args.source == "dolly":
        questions = prepare_questions_from_dolly(
            num_questions=args.num_questions,
            seed=args.seed,
            output_file=args.output_file
        )
    elif args.source == "existing":
        if not args.input_file:
            raise ValueError("--input-file is required when source='existing'")
        
        questions = prepare_questions_from_existing(
            input_file=args.input_file,
            output_file=args.output_file,
            max_questions=args.num_questions
        )
    
    # Create preference format if requested
    if args.create_preference_format:
        preference_file = args.output_file.replace('.json', '_preference.json')
        create_preference_format(questions, preference_file)
    
    # Show sample questions
    print("\nSample questions:")
    print("-" * 40)
    for i, question in enumerate(questions[:3]):
        print(f"{i+1}. {question[:100]}{'...' if len(question) > 100 else ''}")
    
    print(f"\n✅ Question preparation complete!")
    print(f"Questions saved to: {args.output_file}")
    
    if args.create_preference_format:
        preference_file = args.output_file.replace('.json', '_preference.json')
        print(f"Preference format saved to: {preference_file}")


if __name__ == "__main__":
    main()