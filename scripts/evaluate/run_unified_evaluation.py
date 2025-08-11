#!/usr/bin/env python3
"""
Unified evaluation script for all methods.
Compares BON selection methods and generation methods using the same judge.

Usage:
    python scripts/evaluate/run_unified_evaluation.py \
        --methods bon-drift bon-mle qalign-drift drift-decoding \
        --judge golden \
        --n_values 10,50,100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.base_evaluator import BaseEvaluator, EvaluationResult, OracleEvaluator, RandomEvaluator
from src.evaluation.bon_evaluator import BONEvaluator, PreferenceVectorSelector
from src.evaluation.generation_evaluator import (
    GenerationEvaluator,
    QAlignDriftGenerator,
    QAlignMLEGenerator,
    DriftDecodingGenerator
)
from src.evaluation.judges.llm_judge import PersonaJudge
from src.core.drift import DriftLogitsProcessor


def load_bon_data(data_path: str) -> Dict[str, List[str]]:
    """Load BON dataset and organize by prompt."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    bon_dict = {}
    for item in data:
        prompt = item['prompt']
        outputs = item['outputs']
        bon_dict[prompt] = outputs
    
    return bon_dict


def load_models_and_configs(args):
    """Load models and configurations based on arguments."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configurations
    configs = {}
    
    # Load base/attribute prompts if needed
    if any('drift' in m or 'mle' in m for m in args.methods):
        from src.core.attribute_prompts import attribute_prompts, base_prompt
        configs['base_prompt'] = base_prompt
        configs['attribute_prompts'] = attribute_prompts
    
    # Load models based on methods
    models = {}
    
    if any('drift' in m for m in args.methods):
        # Load drift model and p vector
        if args.drift_model_path:
            from vllm import LLM
            models['drift_model'] = LLM(
                model=args.drift_model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5
            )
        
        if args.p_vector_path:
            with open(args.p_vector_path, 'r') as f:
                p_data = json.load(f)
                # Extract p vector (adjust based on your format)
                if isinstance(p_data, list):
                    configs['p_vector'] = np.array(p_data[0]['p'])
                else:
                    configs['p_vector'] = np.array(p_data['p'])
    
    if any('mle' in m for m in args.methods):
        # Load MLE model and p vector
        if args.mle_p_vector_path:
            with open(args.mle_p_vector_path, 'r') as f:
                mle_data = json.load(f)
                configs['p_vector_mle'] = np.array(mle_data['p'])
    
    # Load base model for generation if needed
    if any('qalign' in m or 'decoding' in m for m in args.methods):
        if args.base_model_path:
            models['base_model'] = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto"
            )
            models['tokenizer'] = AutoTokenizer.from_pretrained(args.base_model_path)
            models['tokenizer'].pad_token = models['tokenizer'].eos_token
    
    configs['device'] = device
    return models, configs


def create_evaluators(args, judge, bon_data, models, configs) -> Dict[str, BaseEvaluator]:
    """Create evaluator instances for selected methods."""
    
    evaluators = {}
    device = configs['device']
    
    # Always include baselines
    evaluators['Random'] = RandomEvaluator(judge, bon_data, seed=42)
    evaluators['Oracle'] = OracleEvaluator(judge, bon_data)
    
    # BON Methods
    if 'bon-drift' in args.methods:
        if 'drift_model' in models and 'p_vector' in configs:
            selector = PreferenceVectorSelector(
                models['drift_model'],
                configs['p_vector'],
                configs['base_prompt'],
                configs['attribute_prompts'],
                models.get('tokenizer'),
                device
            )
            evaluators['BON-Drift'] = BONEvaluator(
                judge, selector, bon_data, method_name="BON-Drift"
            )
    
    if 'bon-mle' in args.methods:
        if 'p_vector_mle' in configs:
            selector = PreferenceVectorSelector(
                models.get('drift_model'),  # Can reuse drift model
                configs['p_vector_mle'],
                configs['base_prompt'],
                configs['attribute_prompts'],
                models.get('tokenizer'),
                device
            )
            evaluators['BON-MLE'] = BONEvaluator(
                judge, selector, bon_data, method_name="BON-MLE"
            )
    
    
    # Generation Methods
    if 'qalign-drift' in args.methods:
        if 'base_model' in models and 'drift_model' in models and 'p_vector' in configs:
            generator = QAlignDriftGenerator(
                models['base_model'],
                models['drift_model'],
                configs['p_vector'],
                configs['base_prompt'],
                configs['attribute_prompts'],
                models['tokenizer'],
                num_samples=args.qalign_samples,
                temperature=args.temperature,
                max_length=args.max_length,
                device=device
            )
            evaluators['QAlign-Drift'] = GenerationEvaluator(
                judge, generator, method_name="QAlign-Drift"
            )
    
    if 'qalign-mle' in args.methods:
        if 'base_model' in models and 'p_vector_mle' in configs:
            generator = QAlignMLEGenerator(
                models['base_model'],
                models.get('drift_model'),
                configs['p_vector_mle'],
                configs['base_prompt'],
                configs['attribute_prompts'],
                models['tokenizer'],
                num_samples=args.qalign_samples,
                temperature=args.temperature,
                max_length=args.max_length,
                device=device
            )
            evaluators['QAlign-MLE'] = GenerationEvaluator(
                judge, generator, method_name="QAlign-MLE"
            )
    
    if 'drift-decoding' in args.methods:
        if 'base_model' in models and 'drift_model' in models and 'p_vector' in configs:
            # Create drift logits processor
            drift_processor = DriftLogitsProcessor(
                b=args.drift_b,
                small_model=models['drift_model'],
                tokenizer=models['tokenizer'],
                base_prompt=configs['base_prompt'],
                attribute_prompts=configs['attribute_prompts'],
                weights=configs['p_vector'].tolist()
            )
            
            generator = DriftDecodingGenerator(
                models['base_model'],
                drift_processor,
                models['tokenizer'],
                max_length=args.max_length,
                temperature=args.temperature,
                device=device
            )
            evaluators['Drift-Decoding'] = GenerationEvaluator(
                judge, generator, method_name="Drift-Decoding"
            )
    
    return evaluators


def run_evaluations(
    evaluators: Dict[str, BaseEvaluator],
    prompts: List[str],
    n_values: List[int],
    output_dir: str
) -> Dict[str, List[EvaluationResult]]:
    """Run evaluations for all methods and n values."""
    
    results = {}
    
    for method_name, evaluator in evaluators.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*70}")
        
        method_results = []
        
        # For BON methods, evaluate with different n values
        if isinstance(evaluator, (BONEvaluator, OracleEvaluator, RandomEvaluator)):
            for n in n_values:
                print(f"\n--- Evaluating with n={n} ---")
                result = evaluator.evaluate(prompts, n=n)
                result.metadata['n'] = n
                method_results.append(result)
                
                # Save individual result
                result_path = f"{output_dir}/{method_name}_n{n}.json"
                evaluator.save_results(result, result_path)
                
                print(result.summary())
        
        # For generation methods, only evaluate once
        else:
            result = evaluator.evaluate(prompts)
            method_results.append(result)
            
            # Save result
            result_path = f"{output_dir}/{method_name}.json"
            evaluator.save_results(result, result_path)
            
            print(result.summary())
        
        results[method_name] = method_results
    
    return results


def print_comparison_table(results: Dict[str, List[EvaluationResult]], n_values: List[int]):
    """Print comparison table of all methods."""
    
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    
    # For BON methods with multiple n values
    bon_methods = ['Random', 'Oracle', 'BON-Drift', 'BON-MLE', 'BON-Persona']
    header = f"{'Method':<20}"
    for n in n_values:
        header += f"{'n='+str(n):<15}"
    print(header)
    print("-"*100)
    
    for method in bon_methods:
        if method in results:
            row = f"{method:<20}"
            for result in results[method]:
                row += f"{result.mean_score:.3f}±{result.std_score:.2f}  "
            print(row)
    
    # For generation methods
    print("\nGeneration Methods:")
    print("-"*50)
    gen_methods = ['QAlign-Drift', 'QAlign-MLE', 'Drift-Decoding']
    for method in gen_methods:
        if method in results:
            result = results[method][0]
            print(f"{method:<20} {result.mean_score:.3f}±{result.std_score:.2f}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for all methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Method selection
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bon-drift", "random", "oracle"],
        choices=[
            "bon-drift", "bon-mle",
            "qalign-drift", "qalign-mle", 
            "drift-decoding",
            "random", "oracle"
        ],
        help="Methods to evaluate"
    )
    
    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/bon.json",
        help="Path to BON dataset"
    )
    
    parser.add_argument(
        "--test_prompts",
        type=str,
        default=None,
        help="Path to test prompts (if different from BON data)"
    )
    
    # Judge configuration
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="LLM model to use as judge"
    )
    
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="VLLM endpoint URL for judge"
    )
    
    parser.add_argument(
        "--judge_cache_dir",
        type=str,
        default="cache/persona_judge",
        help="Directory to cache judge responses"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        default="A helpful assistant",
        help="Persona description for evaluation"
    )
    
    # Model paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model for generation"
    )
    
    parser.add_argument(
        "--drift_model_path",
        type=str,
        default=None,
        help="Path to drift model"
    )
    
    parser.add_argument(
        "--p_vector_path",
        type=str,
        default=None,
        help="Path to p vector for drift"
    )
    
    parser.add_argument(
        "--mle_p_vector_path",
        type=str,
        default=None,
        help="Path to MLE-optimized p vector"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n_values",
        type=str,
        default="10,50,100",
        help="Comma-separated n values for BON"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate"
    )
    
    # Generation parameters
    parser.add_argument(
        "--qalign_samples",
        type=int,
        default=32,
        help="Number of samples for QAlign"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--drift_b",
        type=float,
        default=1.0,
        help="Drift decoding parameter b"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/unified_evaluation",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Parse n values
    n_values = [int(x) for x in args.n_values.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load BON data
    print("Loading BON data...")
    bon_data = load_bon_data(args.data_path)
    prompts = list(bon_data.keys())
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize judge
    print(f"\nInitializing LLM judge ({args.judge_model})...")
    persona_judge = PersonaJudge(
        base_url=args.judge_base_url,
        model=args.judge_model,
        cache_dir=args.judge_cache_dir,
        temperature=0.1,  # Low temperature for consistent scoring
        max_tokens=512
    )
    
    # Create judge wrapper with persona
    class PersonaBoundJudge:
        def __init__(self, persona_judge, persona):
            self.persona_judge = persona_judge
            self.persona = persona
        
        def score(self, prompt, response):
            return self.persona_judge.score(prompt, response, self.persona)
        
        def get_statistics(self):
            return self.persona_judge.get_statistics()
    
    judge = PersonaBoundJudge(persona_judge, args.persona)
    print(f"Using persona: {args.persona}")
    print(f"Judge endpoint: {args.judge_base_url}")
    
    # Load models and configs
    print("\nLoading models and configurations...")
    models, configs = load_models_and_configs(args)
    
    # Create evaluators
    print("\nCreating evaluators...")
    evaluators = create_evaluators(args, judge, bon_data, models, configs)
    print(f"Created {len(evaluators)} evaluators: {list(evaluators.keys())}")
    
    # Run evaluations
    print("\nRunning evaluations...")
    results = run_evaluations(evaluators, prompts, n_values, str(output_dir))
    
    # Print comparison
    print_comparison_table(results, n_values)
    
    # Save combined results
    combined_results = {}
    for method, method_results in results.items():
        combined_results[method] = [r.to_dict() for r in method_results]
    
    combined_path = output_dir / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to: {combined_path}")
    
    # Print judge statistics if available
    if hasattr(judge, 'get_statistics'):
        stats = judge.get_statistics()
        print(f"\nJudge Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()