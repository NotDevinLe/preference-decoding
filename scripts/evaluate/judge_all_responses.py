#!/usr/bin/env python3
"""
Unified judging pipeline for all method responses.
Judges all responses using the same PersonaJudge to ensure fair comparison.

Usage:
    python scripts/evaluate/judge_all_responses.py \
        --config configs/experiment_config.yaml \
        --prompts data/processed/evaluation_prompts.json \
        --responses_dir results/responses \
        --output_dir results/judgments
"""

import os
import sys
import json
import yaml
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.judges.llm_judge import PersonaJudge
from src.evaluation.judges.persona_rubric import PersonaScore, extract_persona_from_prompt


@dataclass
class JudgedResponse:
    """Container for a judged response."""
    method: str
    prompt_id: int
    prompt: str
    persona: str
    response: str
    scores: Dict[str, float]
    reasoning: Dict[str, str]
    overall_score: float
    timestamp: float


class UnifiedJudgingPipeline:
    """Pipeline for judging all method responses with consistent criteria."""
    
    def __init__(self, config: dict):
        """
        Initialize judging pipeline.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.judge_config = config['judge']
        
        # Initialize judge
        self._initialize_judge()
        
        # Track statistics
        self.stats = {
            'total_judgments': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'failures': 0
        }
        
        print(f"Initialized judging pipeline with {self.judge_config['type']} judge")
    
    def _initialize_judge(self):
        """Initialize the appropriate judge."""
        if self.judge_config['type'] == 'persona':
            judge_settings = self.judge_config['persona_judge']
            
            self.judge = PersonaJudge(
                cache_dir=judge_settings['cache_dir'],
                temperature=judge_settings['temperature'],
                max_tokens=judge_settings['max_tokens'],
                max_retries=judge_settings['max_retries']
            )
        else:
            raise ValueError(f"Unknown judge type: {self.judge_config['type']}")
        
        print(f"Judge initialized: {type(self.judge).__name__}")
    
    def load_prompts(self, prompts_path: str) -> Dict[int, Dict]:
        """Load evaluation prompts indexed by eval_id."""
        with open(prompts_path, 'r') as f:
            data = json.load(f)
        
        prompts_dict = {}
        for prompt in data['prompts']:
            prompts_dict[prompt['eval_id']] = prompt
        
        return prompts_dict
    
    def discover_responses(self, responses_dir: str) -> Dict[str, List[Dict]]:
        """
        Discover all response files organized by method.
        
        Args:
            responses_dir: Directory containing method subdirectories with responses
            
        Returns:
            Dictionary mapping method names to lists of response files
        """
        responses_path = Path(responses_dir)
        method_responses = defaultdict(list)
        
        if not responses_path.exists():
            print(f"Responses directory not found: {responses_dir}")
            return {}
        
        # Look for method subdirectories
        for method_dir in responses_path.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                
                # Find response files in this method directory
                for response_file in method_dir.glob("*.json"):
                    try:
                        with open(response_file, 'r') as f:
                            response_data = json.load(f)
                        
                        # Add method and file info
                        response_data['method'] = method_name
                        response_data['file_path'] = str(response_file)
                        
                        method_responses[method_name].append(response_data)
                    
                    except Exception as e:
                        print(f"Error loading {response_file}: {e}")
        
        # Sort responses by prompt_id for consistent processing
        for method in method_responses:
            method_responses[method].sort(key=lambda x: x.get('prompt_id', 0))
        
        print(f"Discovered responses for {len(method_responses)} methods:")
        for method, responses in method_responses.items():
            print(f"  {method}: {len(responses)} responses")
        
        return method_responses
    
    def judge_response(
        self, 
        prompt: str, 
        persona: str, 
        response: str
    ) -> Optional[PersonaScore]:
        """
        Judge a single response.
        
        Args:
            prompt: The original prompt
            persona: The persona to evaluate against
            response: The response to judge
            
        Returns:
            PersonaScore object or None if judging fails
        """
        try:
            score = self.judge.score_response(persona, prompt, response)
            self.stats['total_judgments'] += 1
            return score
        
        except Exception as e:
            print(f"Error judging response: {e}")
            self.stats['failures'] += 1
            return None
    
    def judge_all_responses(
        self,
        method_responses: Dict[str, List[Dict]],
        prompts_dict: Dict[int, Dict],
        max_responses: Optional[int] = None
    ) -> Dict[str, List[JudgedResponse]]:
        """
        Judge all responses from all methods.
        
        Args:
            method_responses: Response data by method
            prompts_dict: Prompt data indexed by eval_id
            max_responses: Maximum responses per method (for testing)
            
        Returns:
            Dictionary of judged responses by method
        """
        judged_responses = defaultdict(list)
        
        # Process each method
        for method_name, responses in method_responses.items():
            print(f"\n{'='*60}")
            print(f"Judging responses for: {method_name}")
            print(f"{'='*60}")
            
            # Limit responses if specified
            if max_responses:
                responses = responses[:max_responses]
            
            method_judged = []
            
            # Judge each response
            for response_data in tqdm(responses, desc=f"Judging {method_name}"):
                try:
                    prompt_id = response_data.get('prompt_id')
                    
                    # Get prompt data
                    if prompt_id not in prompts_dict:
                        print(f"Warning: Prompt ID {prompt_id} not found in prompts")
                        continue
                    
                    prompt_info = prompts_dict[prompt_id]
                    prompt_text = prompt_info['prompt']
                    persona = prompt_info.get('persona', 'A helpful assistant')
                    
                    # Get response text
                    response_text = response_data.get('response', '')
                    
                    if not response_text.strip():
                        print(f"Warning: Empty response for prompt {prompt_id} in {method_name}")
                        continue
                    
                    # Judge the response
                    score = self.judge_response(prompt_text, persona, response_text)
                    
                    if score is not None:
                        # Create judged response object
                        judged = JudgedResponse(
                            method=method_name,
                            prompt_id=prompt_id,
                            prompt=prompt_text,
                            persona=persona,
                            response=response_text,
                            scores={
                                'speaking_style': score.speaking_style,
                                'personality': score.personality,
                                'knowledge': score.knowledge,
                                'behavioral': score.behavioral,
                                'emotional': score.emotional
                            },
                            reasoning={
                                'speaking_style': score.speaking_reason,
                                'personality': score.personality_reason,
                                'knowledge': score.knowledge_reason,
                                'behavioral': score.behavioral_reason,
                                'emotional': score.emotional_reason
                            },
                            overall_score=score.get_overall(),
                            timestamp=time.time()
                        )
                        
                        method_judged.append(judged)
                
                except Exception as e:
                    print(f"Error processing response {prompt_id} for {method_name}: {e}")
                    continue
            
            judged_responses[method_name] = method_judged
            
            # Print method summary
            if method_judged:
                scores = [j.overall_score for j in method_judged]
                print(f"{method_name} summary:")
                print(f"  Responses judged: {len(method_judged)}")
                print(f"  Average overall score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        return judged_responses
    
    def save_judgments(
        self,
        judged_responses: Dict[str, List[JudgedResponse]],
        output_dir: str
    ):
        """
        Save judgment results to files.
        
        Args:
            judged_responses: Judged responses by method
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual method results
        for method_name, judged_list in judged_responses.items():
            method_file = output_path / f"{method_name}_judgments.jsonl"
            
            with open(method_file, 'w') as f:
                for judged in judged_list:
                    f.write(json.dumps(asdict(judged)) + '\n')
            
            print(f"Saved {len(judged_list)} judgments for {method_name} to {method_file}")
        
        # Save combined results
        combined_file = output_path / "all_judgments.jsonl"
        with open(combined_file, 'w') as f:
            for method_name, judged_list in judged_responses.items():
                for judged in judged_list:
                    f.write(json.dumps(asdict(judged)) + '\n')
        
        print(f"Saved combined judgments to {combined_file}")
        
        # Save summary statistics
        summary = self.compute_summary_stats(judged_responses)
        summary_file = output_path / "judgment_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved summary statistics to {summary_file}")
    
    def compute_summary_stats(
        self, 
        judged_responses: Dict[str, List[JudgedResponse]]
    ) -> Dict[str, Any]:
        """Compute summary statistics across all methods."""
        summary = {
            'total_responses_judged': sum(len(responses) for responses in judged_responses.values()),
            'methods': {},
            'dimension_correlations': {},
            'judge_stats': self.stats.copy()
        }
        
        # Compute per-method statistics
        for method_name, judged_list in judged_responses.items():
            if not judged_list:
                continue
            
            # Extract scores by dimension
            scores_by_dim = defaultdict(list)
            for judged in judged_list:
                for dim, score in judged.scores.items():
                    scores_by_dim[dim].append(score)
                scores_by_dim['overall'].append(judged.overall_score)
            
            # Compute statistics
            method_stats = {}
            for dim, scores in scores_by_dim.items():
                if scores:
                    method_stats[dim] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores)),
                        'median': float(np.median(scores))
                    }
            
            summary['methods'][method_name] = {
                'num_responses': len(judged_list),
                'dimension_stats': method_stats
            }
        
        # Add judge-specific stats
        if hasattr(self.judge, 'get_statistics'):
            judge_stats = self.judge.get_statistics()
            summary['judge_stats'].update(judge_stats)
        
        return summary
    
    def print_comparison_table(self, judged_responses: Dict[str, List[JudgedResponse]]):
        """Print a comparison table of all methods."""
        print("\n" + "="*100)
        print("METHOD COMPARISON TABLE")
        print("="*100)
        
        # Collect data
        methods_data = {}
        for method, judged_list in judged_responses.items():
            if judged_list:
                overall_scores = [j.overall_score for j in judged_list]
                methods_data[method] = {
                    'count': len(judged_list),
                    'mean': np.mean(overall_scores),
                    'std': np.std(overall_scores)
                }
        
        # Print table
        print(f"{'Method':<20} {'Count':<8} {'Mean Score':<12} {'Std Dev':<10}")
        print("-" * 50)
        
        # Sort by mean score (descending)
        sorted_methods = sorted(methods_data.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for method, data in sorted_methods:
            print(f"{method:<20} {data['count']:<8} {data['mean']:<12.3f} {data['std']:<10.3f}")
        
        print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Judge all method responses with unified pipeline"
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
        "--responses_dir",
        type=str,
        default="results/responses",
        help="Directory containing method response subdirectories"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/judgments",
        help="Directory to save judgment results"
    )
    
    parser.add_argument(
        "--max_responses",
        type=int,
        default=None,
        help="Maximum responses per method (for testing)"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Specific methods to judge (default: all found)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("UNIFIED RESPONSE JUDGING")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Prompts: {args.prompts}")
    print(f"Responses dir: {args.responses_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.max_responses:
        print(f"Max responses per method: {args.max_responses}")
    print("="*70)
    
    # Initialize pipeline
    pipeline = UnifiedJudgingPipeline(config)
    
    # Load prompts
    print("Loading prompts...")
    prompts_dict = pipeline.load_prompts(args.prompts)
    print(f"Loaded {len(prompts_dict)} prompts")
    
    # Discover responses
    print("Discovering responses...")
    method_responses = pipeline.discover_responses(args.responses_dir)
    
    if not method_responses:
        print("No responses found. Exiting.")
        return
    
    # Filter methods if specified
    if args.methods:
        filtered_responses = {m: method_responses[m] for m in args.methods if m in method_responses}
        method_responses = filtered_responses
        print(f"Filtered to methods: {list(method_responses.keys())}")
    
    # Judge all responses
    print("\nStarting judgment process...")
    start_time = time.time()
    
    judged_responses = pipeline.judge_all_responses(
        method_responses,
        prompts_dict,
        max_responses=args.max_responses
    )
    
    judgment_time = time.time() - start_time
    
    # Save results
    print("\nSaving judgment results...")
    pipeline.save_judgments(judged_responses, args.output_dir)
    
    # Print final comparison
    pipeline.print_comparison_table(judged_responses)
    
    # Print final summary
    total_judged = sum(len(responses) for responses in judged_responses.values())
    
    print(f"\n{'='*70}")
    print("JUDGING COMPLETE")
    print(f"{'='*70}")
    print(f"Total responses judged: {total_judged}")
    print(f"Methods processed: {len(judged_responses)}")
    print(f"Judgment time: {judgment_time:.1f}s")
    if judgment_time > 0:
        print(f"Responses per second: {total_judged/judgment_time:.2f}")
    print(f"Results saved to: {args.output_dir}")
    
    # Print judge statistics
    print(f"\nJudge Statistics:")
    stats = pipeline.stats
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if hasattr(pipeline.judge, 'get_statistics'):
        judge_stats = pipeline.judge.get_statistics()
        for key, value in judge_stats.items():
            print(f"  {key}: {value}")
    
    print("="*70)


if __name__ == "__main__":
    main()