#!/usr/bin/env python3
"""
Generate shared BON outputs that will be reused by BON-Drift and BON-MLE methods.
This script generates 16 outputs per prompt once, which are then used by both methods.

Usage:
    python scripts/generate/generate_shared_bon.py \
        --config configs/experiment_config.yaml \
        --prompts data/processed/evaluation_prompts.json \
        --output_dir results/bon_outputs
"""

import os
import sys
import json
import yaml
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> List[Dict]:
    """Load evaluation prompts."""
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    return data['prompts']


class BONOutputGenerator:
    """Generator for shared BON outputs."""
    
    def __init__(self, config: dict):
        """
        Initialize BON output generator.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model_config = config['models']
        self.gen_config = config['generation']
        
        # Initialize model
        self._initialize_model()
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.gen_config['bon_temperature'],
            top_p=self.gen_config['bon_top_p'],
            max_tokens=self.gen_config['bon_max_tokens'],
            n=self.gen_config['bon_outputs_per_prompt'],  # Generate N outputs per prompt
            use_beam_search=False
        )
        
        print(f"Initialized BON generator with {self.gen_config['bon_outputs_per_prompt']} outputs per prompt")
    
    def _initialize_model(self):
        """Initialize the language model."""
        model_name = self.model_config['base_model']
        print(f"Loading model: {model_name}")
        
        # Use VLLM for efficient batch generation
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=self.model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.8),
            max_model_len=self.model_config.get('max_model_len', 4096),
            trust_remote_code=True
        )
        
        # Load tokenizer for formatting
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model initialized successfully")
    
    def format_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Format prompt for generation.
        
        Args:
            prompt_data: Prompt data with text
            
        Returns:
            Formatted prompt string
        """
        prompt_text = prompt_data['prompt']
        
        # Create chat format
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return formatted
    
    def generate_outputs(self, prompts: List[Dict], batch_size: int = 10) -> Dict[int, List[str]]:
        """
        Generate BON outputs for all prompts.
        
        Args:
            prompts: List of prompt dictionaries
            batch_size: Number of prompts to process at once
            
        Returns:
            Dictionary mapping prompt eval_id to list of generated outputs
        """
        all_outputs = {}
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating BON outputs"):
            batch = prompts[i:i + batch_size]
            
            # Format prompts
            formatted_prompts = [self.format_prompt(prompt) for prompt in batch]
            
            # Generate outputs
            try:
                outputs = self.model.generate(formatted_prompts, self.sampling_params, use_tqdm=False)
                
                # Extract and store outputs
                for j, output in enumerate(outputs):
                    prompt_id = batch[j]['eval_id']
                    generated_texts = [o.text.strip() for o in output.outputs]
                    
                    # Filter empty or very short outputs
                    valid_outputs = [
                        text for text in generated_texts 
                        if len(text.strip()) >= self.config['quality_control']['min_response_length']
                    ]
                    
                    # Pad with empty strings if not enough valid outputs
                    while len(valid_outputs) < self.gen_config['bon_outputs_per_prompt']:
                        valid_outputs.append("[Generation failed]")
                    
                    all_outputs[prompt_id] = valid_outputs[:self.gen_config['bon_outputs_per_prompt']]
                
            except Exception as e:
                print(f"Error generating batch {i//batch_size + 1}: {e}")
                # Add empty outputs for failed batch
                for prompt in batch:
                    prompt_id = prompt['eval_id']
                    all_outputs[prompt_id] = ["[Generation failed]"] * self.gen_config['bon_outputs_per_prompt']
        
        return all_outputs
    
    def save_outputs(self, outputs: Dict[int, List[str]], output_dir: str, prompts: List[Dict]):
        """
        Save generated outputs to individual files.
        
        Args:
            outputs: Generated outputs by prompt ID
            output_dir: Directory to save outputs
            prompts: Original prompt data for metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create prompt ID to prompt data mapping
        prompt_map = {p['eval_id']: p for p in prompts}
        
        # Save individual files
        for prompt_id, generated_outputs in outputs.items():
            prompt_data = prompt_map[prompt_id]
            
            output_data = {
                'prompt_id': prompt_id,
                'prompt': prompt_data['prompt'],
                'category': prompt_data.get('category', ''),
                'generation_config': {
                    'temperature': self.gen_config['bon_temperature'],
                    'top_p': self.gen_config['bon_top_p'],
                    'max_tokens': self.gen_config['bon_max_tokens'],
                    'num_outputs': self.gen_config['bon_outputs_per_prompt']
                },
                'outputs': generated_outputs,
                'timestamp': time.time()
            }
            
            # Save to file
            output_file = output_path / f"prompt_{prompt_id:04d}_outputs.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(outputs)} BON output files to {output_dir}")
        
        # Save summary metadata
        summary = {
            'total_prompts': len(outputs),
            'outputs_per_prompt': self.gen_config['bon_outputs_per_prompt'],
            'generation_config': {
                'model': self.model_config['base_model'],
                'temperature': self.gen_config['bon_temperature'],
                'top_p': self.gen_config['bon_top_p'],
                'max_tokens': self.gen_config['bon_max_tokens']
            },
            'quality_stats': {
                'total_outputs': len(outputs) * self.gen_config['bon_outputs_per_prompt'],
                'failed_generations': sum(
                    sum(1 for out in outputs_list if "[Generation failed]" in out)
                    for outputs_list in outputs.values()
                )
            },
            'timestamp': time.time()
        }
        
        summary_file = output_path / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Generation summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate shared BON outputs for reuse by BON methods"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment configuration"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/processed/evaluation_prompts.json",
        help="Path to evaluation prompts"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bon_outputs",
        help="Directory to save BON outputs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for generation (overrides config)"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (for testing)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set batch size
    batch_size = args.batch_size or config['compute']['batch_size']
    
    print("="*70)
    print("SHARED BON OUTPUT GENERATION")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Prompts: {args.prompts}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {batch_size}")
    print("="*70)
    
    # Load prompts
    print("Loading prompts...")
    prompts = load_prompts(args.prompts)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Limited to {args.max_prompts} prompts")
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Check for existing outputs if resuming
    existing_outputs = set()
    if args.resume:
        output_path = Path(args.output_dir)
        if output_path.exists():
            existing_files = list(output_path.glob("prompt_*_outputs.json"))
            existing_outputs = {
                int(f.stem.split('_')[1]) for f in existing_files
                if f.stem.startswith('prompt_') and f.stem.endswith('_outputs')
            }
            print(f"Found {len(existing_outputs)} existing outputs, will skip these")
    
    # Filter prompts to generate
    if existing_outputs:
        prompts_to_generate = [p for p in prompts if p['eval_id'] not in existing_outputs]
        print(f"Will generate outputs for {len(prompts_to_generate)} prompts")
    else:
        prompts_to_generate = prompts
    
    if not prompts_to_generate:
        print("All prompts already have outputs. Exiting.")
        return
    
    # Initialize generator
    generator = BONOutputGenerator(config)
    
    # Generate outputs
    print("Starting generation...")
    start_time = time.time()
    
    outputs = generator.generate_outputs(prompts_to_generate, batch_size)
    
    generation_time = time.time() - start_time
    
    # Save outputs
    print("Saving outputs...")
    generator.save_outputs(outputs, args.output_dir, prompts_to_generate)
    
    # Print summary
    total_outputs = len(outputs) * config['generation']['bon_outputs_per_prompt']
    outputs_per_second = total_outputs / generation_time if generation_time > 0 else 0
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Prompts processed: {len(outputs)}")
    print(f"Total outputs generated: {total_outputs}")
    print(f"Generation time: {generation_time:.1f}s")
    print(f"Outputs per second: {outputs_per_second:.2f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()