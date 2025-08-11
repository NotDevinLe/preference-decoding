#!/usr/bin/env python3
"""
Generation script for individual methods.
Generates responses using a single specified method and saves outputs to JSON.

Usage:
    python scripts/generate/run_generation.py \
        --method bon-drift \
        --data_path data/bon.json \
        --output_path results/generations/bon-drift.json \
        --drift_model_path models/drift_model \
        --p_vector_path vectors/p_vector.json
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
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.base_evaluator import BaseEvaluator, OracleEvaluator, RandomEvaluator
from src.evaluation.bon_evaluator import BONEvaluator, PreferenceVectorSelector
from src.evaluation.generation_evaluator import (
    GenerationEvaluator,
    DriftDecodingGenerator
)
from src.evaluation.qalign_generator import QAlignGenerator
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


def load_models_and_configs(args, method: str):
    """Load models and configurations for the specified method."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    configs = {'device': device}
    models = {}
    
    # Load base/attribute prompts if needed for preference vector methods
    if 'drift' in method or 'mle' in method:
        from src.core.attribute_prompts import attribute_prompts, base_prompt
        configs['base_prompt'] = base_prompt
        configs['attribute_prompts'] = attribute_prompts
    
    # Load models based on method
    if method in ['bon-drift', 'bon-mle', 'qalign-drift', 'qalign-mle', 'drift-decoding']:
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
                if isinstance(p_data, list):
                    configs['p_vector'] = np.array(p_data[0]['p'])
                    print(f"Loaded p vector: {configs['p_vector']}")
                else:
                    configs['p_vector'] = np.array(p_data['p'])
    
    if method in ['bon-mle', 'qalign-mle']:
        # Load MLE p vector (can use same path as drift if not specified)
        mle_path = args.mle_p_vector_path or args.p_vector_path
        if mle_path:
            with open(mle_path, 'r') as f:
                mle_data = json.load(f)
                if isinstance(mle_data, list):
                    configs['p_vector_mle'] = np.array(mle_data[0]['p'])
                else:
                    configs['p_vector_mle'] = np.array(mle_data['p'])
                print(f"Loaded MLE p vector from {mle_path}: {configs['p_vector_mle']}")
        else:
            raise ValueError(f"Method {method} requires either --mle_p_vector_path or --p_vector_path")
    
    # Load tokenizer for methods that need it
    if method in ['bon-drift', 'bon-mle'] and args.drift_model_path:
        # BON methods need tokenizer for drift scoring
        models['tokenizer'] = AutoTokenizer.from_pretrained(args.drift_model_path)
        models['tokenizer'].pad_token = models['tokenizer'].eos_token
    
    # Load base model for generation methods
    if method.startswith('qalign') or method == 'drift-decoding':
        if args.base_model_path:
            models['base_model'] = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto"
            )
            models['tokenizer'] = AutoTokenizer.from_pretrained(args.base_model_path)
            models['tokenizer'].pad_token = models['tokenizer'].eos_token
    
    return models, configs


def create_generator(method: str, args, bon_data, models, configs):
    """Create generator/evaluator for the specified method."""
    
    device = configs['device']
    
    # Dummy judge for evaluators (we only need generation, not evaluation)
    class DummyJudge:
        def score(self, prompt, response):
            return 1.0  # Dummy score
    
    dummy_judge = DummyJudge()
    
    if method == 'bon-drift':
        selector = PreferenceVectorSelector(
            models['drift_model'],
            configs['p_vector'],
            configs['base_prompt'],
            configs['attribute_prompts'],
            models.get('tokenizer'),
            device
        )
        return BONEvaluator(dummy_judge, selector, bon_data, method_name="BON-Drift")
    
    elif method == 'bon-mle':
        selector = PreferenceVectorSelector(
            models.get('drift_model'),  # Can reuse drift model
            configs['p_vector_mle'],
            configs['base_prompt'],
            configs['attribute_prompts'],
            models.get('tokenizer'),
            device
        )
        return BONEvaluator(dummy_judge, selector, bon_data, method_name="BON-MLE")
    
    elif method == 'qalign-drift':
        generator = QAlignGenerator(
            base_model=models['base_model'],
            scoring_model=models['drift_model'],
            p_vector=configs['p_vector'],
            base_prompt=configs['base_prompt'],
            attribute_prompts=configs['attribute_prompts'],
            tokenizer=models['tokenizer'],
            num_steps=args.qalign_steps,
            beta=args.qalign_beta,
            temperature=args.temperature,
            max_length=args.max_length,
            device=device,
            method_name="QAlign-Drift"
        )
        return GenerationEvaluator(dummy_judge, generator, method_name="QAlign-Drift")
    
    elif method == 'qalign-mle':
        generator = QAlignGenerator(
            base_model=models['base_model'],
            scoring_model=models.get('drift_model'),  # MLE uses same model type
            p_vector=configs['p_vector_mle'],
            base_prompt=configs['base_prompt'],
            attribute_prompts=configs['attribute_prompts'],
            tokenizer=models['tokenizer'],
            num_steps=args.qalign_steps,
            beta=args.qalign_beta,
            temperature=args.temperature,
            max_length=args.max_length,
            device=device,
            method_name="QAlign-MLE"
        )
        return GenerationEvaluator(dummy_judge, generator, method_name="QAlign-MLE")
    
    elif method == 'drift-decoding':
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
        return GenerationEvaluator(dummy_judge, generator, method_name="Drift-Decoding")
    
    elif method == 'random':
        return RandomEvaluator(dummy_judge, bon_data, seed=42)
    
    elif method == 'oracle':
        return OracleEvaluator(dummy_judge, bon_data)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_responses(generator: BaseEvaluator, prompts: List[str], method: str, args) -> List[str]:
    """Generate responses using the specified generator."""
    
    print(f"Generating responses using {method}...")
    start_time = time.time()
    
    # Generate responses
    if method.startswith('bon') or method in ['random', 'oracle']:
        # BON methods need n parameter
        responses = generator.get_responses(prompts, n=args.n_value)
    else:
        # Generation methods
        responses = generator.get_responses(prompts)
    
    generation_time = time.time() - start_time
    print(f"Generated {len(responses)} responses in {generation_time:.2f}s")
    
    return responses


def save_generation_results(
    method: str,
    prompts: List[str],
    responses: List[str], 
    output_path: str,
    args,
    generation_time: float
):
    """Save generation results to JSON file."""
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "generation_time": generation_time,
        "num_prompts": len(prompts),
        "prompts": prompts,
        "responses": responses,
        "parameters": {
            "n_value": getattr(args, 'n_value', None),
            "temperature": getattr(args, 'temperature', None),
            "max_length": getattr(args, 'max_length', None),
            "qalign_samples": getattr(args, 'qalign_samples', None),
            "drift_b": getattr(args, 'drift_b', None),
        },
        "model_paths": {
            "base_model_path": getattr(args, 'base_model_path', None),
            "drift_model_path": getattr(args, 'drift_model_path', None),
            "p_vector_path": getattr(args, 'p_vector_path', None),
            "mle_p_vector_path": getattr(args, 'mle_p_vector_path', None),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using a single method",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "bon-drift", "bon-mle",
            "qalign-drift", "qalign-mle", 
            "drift-decoding",
            "random", "oracle"
        ],
        help="Method to use for generation"
    )
    
    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to BON dataset or prompts"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Path to save generated responses"
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
    
    # Generation parameters
    parser.add_argument(
        "--n_value",
        type=int,
        default=50,
        help="n value for BON methods"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process"
    )
    
    parser.add_argument(
        "--qalign_steps",
        type=int,
        default=100,
        help="Number of MCMC steps for QAlign"
    )
    
    parser.add_argument(
        "--qalign_beta",
        type=float,
        default=1.0,
        help="Temperature parameter (beta) for QAlign acceptance probability"
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
    
    args = parser.parse_args()
    
    # Load BON data
    print("Loading data...")
    bon_data = load_bon_data(args.data_path)
    prompts = list(bon_data.keys())
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Load models and configs for the specified method
    print(f"\nLoading models and configurations for {args.method}...")
    models, configs = load_models_and_configs(args, args.method)
    
    # Create generator
    print(f"\nCreating generator for {args.method}...")
    generator = create_generator(args.method, args, bon_data, models, configs)
    
    # Generate responses
    start_time = time.time()
    responses = generate_responses(generator, prompts, args.method, args)
    generation_time = time.time() - start_time
    
    # Save results
    save_generation_results(
        args.method, 
        prompts, 
        responses, 
        args.output_path,
        args,
        generation_time
    )
    
    print(f"\n✅ Generation complete!")
    print(f"Method: {args.method}")
    print(f"Prompts: {len(prompts)}")
    print(f"Time: {generation_time:.2f}s")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()