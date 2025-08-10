#!/usr/bin/env python3
"""
Main script to run persona-based evaluation on BON dataset.

Usage:
    python scripts/evaluate/run_persona_evaluation.py \
        --data_path data/bon.json \
        --output_path results/evaluations/persona_scores.jsonl \
        --max_outputs 20 \
        --workers 4
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.judges.llm_judge import PersonaJudge
from src.evaluation.judges.persona_rubric import extract_persona_from_prompt
from utils.generate_persona_evaluations import (
    evaluate_bon_dataset,
    print_evaluation_statistics,
    analyze_by_prompt
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate persona evaluations for BON dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate first 10 prompts with 20 outputs each
  %(prog)s --data_path data/bon.json --max_prompts 10 --max_outputs 20
  
  # Use async mode for faster evaluation
  %(prog)s --async_mode --workers 8
  
  # Analyze existing evaluations
  %(prog)s --analyze --output_path results/evaluations/persona_scores.jsonl
        """
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/bon.json",
        help="Path to BON dataset JSON"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluations/persona_scores.jsonl",
        help="Path to save evaluations (JSONL)"
    )
    
    parser.add_argument(
        "--max_outputs",
        type=int,
        default=20,
        help="Maximum outputs to evaluate per prompt"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (None for all)"
    )
    
    parser.add_argument(
        "--async_mode",
        action="store_true",
        help="Use async evaluation for higher throughput"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from existing evaluations"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Override persona for all evaluations"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis on existing evaluations"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.analyze:
        # Just analyze existing evaluations
        print("Loading existing evaluations for analysis...")
        evaluations = []
        with open(args.output_path, 'r') as f:
            for line in f:
                evaluations.append(json.loads(line))
        
        judge = PersonaJudge()
        print_evaluation_statistics(evaluations, judge)
        analyze_by_prompt(evaluations)
    else:
        # Run evaluations
        print("=" * 70)
        print("PERSONA EVALUATION")
        print("=" * 70)
        print(f"Data path: {args.data_path}")
        print(f"Output path: {args.output_path}")
        print(f"Max outputs per prompt: {args.max_outputs}")
        print(f"Max prompts: {args.max_prompts or 'All'}")
        print(f"Async mode: {args.async_mode}")
        print(f"Workers: {args.workers}")
        print(f"Resume: {not args.no_resume}")
        print("=" * 70)
        
        evaluate_bon_dataset(
            data_path=args.data_path,
            output_path=args.output_path,
            max_outputs_per_prompt=args.max_outputs,
            max_prompts=args.max_prompts,
            use_async=args.async_mode,
            max_workers=args.workers,
            resume=not args.no_resume,
            persona_override=args.persona
        )
        
        print("\n✅ Evaluation complete!")
        print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()