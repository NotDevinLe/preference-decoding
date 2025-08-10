#!/usr/bin/env python3
"""
Generate persona evaluations for BON dataset using LLM judge.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

from llm_judge import PersonaJudge, AsyncPersonaJudge
from persona_rubric import PersonaScore, extract_persona_from_prompt
import asyncio


def load_bon_data(data_path: str) -> List[Dict]:
    """Load BON dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def save_evaluations(evaluations: List[Dict], output_path: str):
    """Save evaluations to JSONL file."""
    with open(output_path, 'w') as f:
        for eval_entry in evaluations:
            f.write(json.dumps(eval_entry) + '\n')


def load_existing_evaluations(output_path: str) -> Dict[str, Dict]:
    """Load existing evaluations to resume."""
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Create unique key for each evaluation
                key = f"{entry['prompt']}|||{entry['response']}"
                existing[key] = entry
    return existing


def evaluate_bon_dataset(
    data_path: str,
    output_path: str,
    max_outputs_per_prompt: int = 20,
    max_prompts: Optional[int] = None,
    use_async: bool = False,
    max_workers: int = 4,
    resume: bool = True,
    persona_override: Optional[str] = None
):
    """
    Evaluate BON dataset with persona rubric.
    
    Args:
        data_path: Path to BON dataset JSON
        output_path: Path to save evaluations (JSONL)
        max_outputs_per_prompt: Maximum outputs to evaluate per prompt
        max_prompts: Maximum number of prompts to process (None for all)
        use_async: Use async evaluation for higher throughput
        max_workers: Number of parallel workers
        resume: Resume from existing evaluations
        persona_override: Override persona extraction with fixed persona
    """
    
    # Load data
    print(f"Loading BON data from {data_path}...")
    bon_data = load_bon_data(data_path)
    
    if max_prompts:
        bon_data = bon_data[:max_prompts]
    
    print(f"Loaded {len(bon_data)} prompts")
    print(f"Will evaluate up to {max_outputs_per_prompt} outputs per prompt")
    
    # Load existing evaluations if resuming
    existing_evaluations = {}
    if resume:
        existing_evaluations = load_existing_evaluations(output_path)
        print(f"Found {len(existing_evaluations)} existing evaluations")
    
    # Initialize judge
    judge = PersonaJudge()
    
    # Prepare all evaluations
    all_evaluations = []
    evaluations_to_run = []
    
    for prompt_idx, item in enumerate(bon_data):
        prompt = item['prompt']
        
        # Extract or use override persona
        if persona_override:
            persona = persona_override
        else:
            persona = extract_persona_from_prompt(prompt)
        
        # Limit outputs
        outputs = item['outputs'][:max_outputs_per_prompt]
        
        for output_idx, response in enumerate(outputs):
            # Check if already evaluated
            eval_key = f"{prompt}|||{response}"
            
            if eval_key in existing_evaluations:
                all_evaluations.append(existing_evaluations[eval_key])
            else:
                eval_entry = {
                    'prompt_idx': prompt_idx,
                    'output_idx': output_idx,
                    'prompt': prompt,
                    'persona': persona,
                    'response': response,
                    'key': eval_key
                }
                evaluations_to_run.append(eval_entry)
    
    print(f"Need to run {len(evaluations_to_run)} new evaluations")
    
    if len(evaluations_to_run) == 0:
        print("All evaluations already complete!")
        return
    
    # Run evaluations
    if use_async:
        print("Using async evaluation...")
        # Run async evaluations
        async def run_async_evaluations():
            async_judge = AsyncPersonaJudge()
            
            # Prepare evaluation data
            eval_data = [
                {
                    'persona': e['persona'],
                    'question': e['prompt'],
                    'response': e['response']
                }
                for e in evaluations_to_run
            ]
            
            # Progress callback
            pbar = tqdm(total=len(eval_data), desc="Evaluating")
            
            async def progress_callback(completed, total):
                pbar.update(1)
            
            # Run batch evaluation
            scores = await async_judge.batch_evaluate_async(
                eval_data,
                max_concurrent=max_workers,
                progress_callback=progress_callback
            )
            
            pbar.close()
            return scores
        
        # Run async evaluations
        scores = asyncio.run(run_async_evaluations())
    else:
        print("Using synchronous evaluation...")
        
        # Prepare evaluation data
        eval_data = [
            {
                'persona': e['persona'],
                'question': e['prompt'],
                'response': e['response']
            }
            for e in evaluations_to_run
        ]
        
        # Progress callback
        pbar = tqdm(total=len(eval_data), desc="Evaluating")
        
        def progress_callback(completed, total):
            pbar.update(1)
        
        # Run batch evaluation
        scores = judge.batch_evaluate(
            eval_data,
            max_workers=max_workers,
            progress_callback=progress_callback
        )
        
        pbar.close()
    
    # Process results
    print("\nProcessing results...")
    new_evaluations = []
    
    for eval_entry, score in zip(evaluations_to_run, scores):
        if score is not None:
            result = {
                'prompt_idx': eval_entry['prompt_idx'],
                'output_idx': eval_entry['output_idx'],
                'prompt': eval_entry['prompt'],
                'persona': eval_entry['persona'],
                'response': eval_entry['response'],
                'scores': {
                    'speaking_style': score.speaking_style,
                    'personality': score.personality,
                    'knowledge': score.knowledge,
                    'behavioral': score.behavioral,
                    'emotional': score.emotional,
                    'overall': score.get_overall()
                },
                'reasoning': {
                    'speaking_style': score.speaking_reason,
                    'personality': score.personality_reason,
                    'knowledge': score.knowledge_reason,
                    'behavioral': score.behavioral_reason,
                    'emotional': score.emotional_reason
                },
                'timestamp': time.time()
            }
            new_evaluations.append(result)
            all_evaluations.append(result)
    
    # Save all evaluations
    print(f"\nSaving {len(new_evaluations)} new evaluations to {output_path}...")
    
    # Append new evaluations
    with open(output_path, 'a') as f:
        for eval_entry in new_evaluations:
            f.write(json.dumps(eval_entry) + '\n')
    
    # Print statistics
    print_evaluation_statistics(all_evaluations, judge)


def print_evaluation_statistics(evaluations: List[Dict], judge: PersonaJudge):
    """Print statistics about the evaluations."""
    
    if not evaluations:
        print("No evaluations to analyze")
        return
    
    # Extract scores
    all_scores = {
        'speaking_style': [],
        'personality': [],
        'knowledge': [],
        'behavioral': [],
        'emotional': [],
        'overall': []
    }
    
    for eval_entry in evaluations:
        if 'scores' in eval_entry:
            for key in all_scores:
                if key in eval_entry['scores']:
                    all_scores[key].append(eval_entry['scores'][key])
    
    print("\n=== EVALUATION STATISTICS ===")
    print(f"Total evaluations: {len(evaluations)}")
    
    print("\nScore distributions:")
    for dimension, scores in all_scores.items():
        if scores:
            scores_array = np.array(scores)
            print(f"\n{dimension.replace('_', ' ').title()}:")
            print(f"  Mean: {scores_array.mean():.2f}")
            print(f"  Std:  {scores_array.std():.2f}")
            print(f"  Min:  {scores_array.min()}")
            print(f"  Max:  {scores_array.max()}")
            
            # Distribution
            unique, counts = np.unique(scores_array.astype(int), return_counts=True)
            print(f"  Distribution:")
            for score, count in zip(unique, counts):
                pct = (count / len(scores)) * 100
                print(f"    {score}: {count} ({pct:.1f}%)")
    
    # Judge statistics
    print(f"\n=== JUDGE STATISTICS ===")
    stats = judge.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Cache efficiency
    if stats['total_calls'] > 0:
        cache_rate = (stats['cache_hits'] / stats['total_calls']) * 100
        print(f"Cache hit rate: {cache_rate:.1f}%")


def analyze_by_prompt(evaluations: List[Dict]):
    """Analyze evaluations grouped by prompt."""
    
    # Group by prompt
    prompt_groups = {}
    for eval_entry in evaluations:
        prompt = eval_entry['prompt']
        if prompt not in prompt_groups:
            prompt_groups[prompt] = []
        prompt_groups[prompt].append(eval_entry)
    
    print(f"\n=== ANALYSIS BY PROMPT ({len(prompt_groups)} prompts) ===")
    
    for prompt, evals in list(prompt_groups.items())[:5]:  # Show first 5
        print(f"\nPrompt: {prompt[:100]}...")
        print(f"Persona: {evals[0]['persona'][:100]}...")
        print(f"Number of responses evaluated: {len(evals)}")
        
        # Get best and worst responses
        evals_sorted = sorted(evals, key=lambda x: x['scores']['overall'])
        
        if len(evals_sorted) > 0:
            worst = evals_sorted[0]
            best = evals_sorted[-1]
            
            print(f"\nWorst response (score: {worst['scores']['overall']:.2f}):")
            print(f"  {worst['response'][:200]}...")
            
            print(f"\nBest response (score: {best['scores']['overall']:.2f}):")
            print(f"  {best['response'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate persona evaluations for BON dataset")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/bon.json",
        help="Path to BON dataset JSON"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="../results/persona_evaluations_bon.jsonl",
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