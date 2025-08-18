#!/usr/bin/env python3
"""
Compare pre-generated responses from multiple methods using LLM judge.
Loads JSON files with generated responses and compares them side-by-side.

Usage:
    python scripts/evaluate/compare_generated_responses.py \
        --response_files results/bon_drift_responses.json results/qalign_drift_responses.json \
        --method_names "BON-Drift" "QAlign-Drift" \
        --judge_model meta-llama/Llama-3.3-70B-Instruct \
        --judge_base_url http://localhost:8000/v1
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class VLLMJudge:
    """LLM Judge using online VLLM serving."""
    
    def __init__(
        self, 
        base_url: str,
        model: str,
        cache_dir: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Simple cache to avoid re-evaluating identical comparisons
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0
        }
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        cache_file = self.cache_dir / "judge_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached judgments")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.cache_dir:
            cache_file = self.cache_dir / "judge_cache.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    def _create_cache_key(self, prompt: str, responses: List[str], method_names: List[str]) -> str:
        """Create a cache key for the comparison."""
        import hashlib
        
        # Sort by method name to make cache key consistent
        sorted_pairs = sorted(zip(method_names, responses))
        content = prompt + "|".join(f"{name}:{resp}" for name, resp in sorted_pairs)
        return hashlib.md5(content.encode()).hexdigest()
    
    def compare_responses(
        self,
        prompt: str,
        responses: List[str],
        method_names: List[str],
        persona: str = "A helpful and harmless assistant"
    ) -> Dict[str, Any]:
        """
        Compare multiple responses to a single prompt.
        
        Returns:
            Dict with rankings, scores, and reasoning
        """
        self.stats["total_requests"] += 1
        
        # Check cache
        cache_key = self._create_cache_key(prompt, responses, method_names)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        # Create comparison prompt
        comparison_prompt = self._create_comparison_prompt(prompt, responses, method_names, persona)
        
        # Call VLLM API
        try:
            result = self._call_vllm_api(comparison_prompt)
            parsed_result = self._parse_judgment(result, method_names)
            
            # Cache the result
            self.cache[cache_key] = parsed_result
            if len(self.cache) % 10 == 0:  # Save cache periodically
                self._save_cache()
            
            return parsed_result
            
        except Exception as e:
            print(f"Error in LLM judgment: {e}")
            # Return neutral scores as fallback
            return {
                "rankings": {name: len(method_names) // 2 for name in method_names},
                "scores": {name: 5.0 for name in method_names},
                "reasoning": f"Error during judgment: {e}",
                "winner": method_names[0] if method_names else "unknown"
            }
    
    def _create_comparison_prompt(
        self,
        prompt: str,
        responses: List[str],
        method_names: List[str],
        persona: str
    ) -> str:
        """Create a prompt for comparing multiple responses."""
        
        # Format responses with labels
        formatted_responses = ""
        for i, (name, response) in enumerate(zip(method_names, responses)):
            formatted_responses += f"\n**Response {chr(65+i)} ({name}):**\n{response}\n"
        
        comparison_prompt = f"""You are an expert evaluator assessing the quality of AI assistant responses. Your task is to compare multiple responses to the same prompt and provide detailed analysis.

**Target Persona:** {persona}

**User Prompt:**
{prompt}

**Responses to Compare:**{formatted_responses}

**Evaluation Criteria:**
1. **Relevance**: How well does the response address the specific question/request?
2. **Accuracy**: Is the information provided factually correct?
3. **Helpfulness**: How useful is the response to the user?
4. **Clarity**: How clear and well-structured is the response?
5. **Completeness**: Does the response fully address the prompt?
6. **Persona Alignment**: How well does the response match the target persona?

**Please provide your evaluation in the following format:**

**Rankings:** (1=best, {len(method_names)}=worst)
{chr(10).join(f"- {name}: [rank]" for name in method_names)}

**Scores:** (1-10 scale)
{chr(10).join(f"- {name}: [score]/10" for name in method_names)}

**Winner:** [method_name]

**Reasoning:**
[Provide detailed explanation for your rankings, highlighting strengths and weaknesses of each response]
"""
        return comparison_prompt
    
    def _call_vllm_api(self, prompt: str) -> str:
        """Call VLLM API for judgment."""
        import requests
        
        self.stats["api_calls"] += 1
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _parse_judgment(self, judgment: str, method_names: List[str]) -> Dict[str, Any]:
        """Parse the LLM's judgment into structured format."""
        
        rankings = {}
        scores = {}
        winner = "unknown"
        reasoning = judgment
        
        lines = judgment.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if "**Rankings:**" in line:
                current_section = "rankings"
                continue
            elif "**Scores:**" in line:
                current_section = "scores"
                continue
            elif "**Winner:**" in line:
                current_section = "winner"
                continue
            elif "**Reasoning:**" in line:
                current_section = "reasoning"
                reasoning = ""
                continue
            
            if current_section == "rankings" and line.startswith("- "):
                try:
                    for method_name in method_names:
                        if method_name in line:
                            # Extract rank number
                            rank_str = line.split(":")[-1].strip()
                            rank = int(''.join(filter(str.isdigit, rank_str)))
                            rankings[method_name] = rank
                            break
                except:
                    pass
            
            elif current_section == "scores" and line.startswith("- "):
                try:
                    for method_name in method_names:
                        if method_name in line:
                            # Extract score
                            score_part = line.split(":")[-1].strip()
                            score_str = score_part.split("/")[0]
                            score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
                            scores[method_name] = score
                            break
                except:
                    pass
            
            elif current_section == "winner":
                for method_name in method_names:
                    if method_name in line:
                        winner = method_name
                        break
            
            elif current_section == "reasoning":
                reasoning += line + "\n"
        
        # Fill in missing rankings/scores with defaults
        for method_name in method_names:
            if method_name not in rankings:
                rankings[method_name] = len(method_names) // 2 + 1
            if method_name not in scores:
                scores[method_name] = 5.0
        
        return {
            "rankings": rankings,
            "scores": scores,
            "reasoning": reasoning.strip(),
            "winner": winner
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get judge usage statistics."""
        return self.stats.copy()


def load_response_file(file_path: str) -> Dict[str, Any]:
    """Load a response file and return its contents."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_responses_from_files(response_files: List[str], method_names: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract responses from multiple files and align them by prompt.
    
    Returns:
        Tuple of (prompts, method_responses) where method_responses[method][i] is the response
        for prompts[i] from that method.
    """
    
    # Load all files
    all_data = {}
    for file_path, method_name in zip(response_files, method_names):
        print(f"Loading {method_name} from {file_path}...")
        data = load_response_file(file_path)
        all_data[method_name] = data
    
    # Extract prompts and responses
    # Assuming format: {"prompts": [...], "responses": [...]} or {"method": "...", "prompts": [...], "responses": [...]}
    
    prompts = None
    method_responses = {}
    
    for method_name, data in all_data.items():
        if "prompts" in data and "responses" in data:
            if prompts is None:
                prompts = data["prompts"]
            elif prompts != data["prompts"]:
                print(f"Warning: Prompts don't match between files. Using intersection.")
                # Find common prompts
                common_prompts = []
                common_indices = []
                for i, prompt in enumerate(prompts):
                    if i < len(data["prompts"]) and data["prompts"][i] == prompt:
                        common_prompts.append(prompt)
                        common_indices.append(i)
                prompts = common_prompts
                # Update existing method responses
                for existing_method in method_responses:
                    method_responses[existing_method] = [method_responses[existing_method][i] for i in common_indices]
            
            method_responses[method_name] = data["responses"][:len(prompts)]
        
        else:
            raise ValueError(f"Invalid format in {file_path}. Expected 'prompts' and 'responses' keys.")
    
    if prompts is None:
        raise ValueError("No valid prompts found in any file.")
    
    print(f"Found {len(prompts)} common prompts across all methods")
    return prompts, method_responses


def run_comparisons(
    prompts: List[str],
    method_responses: Dict[str, List[str]],
    method_names: List[str],
    judge: VLLMJudge,
    persona: str = "A helpful and harmless assistant"
) -> List[Dict[str, Any]]:
    """Run pairwise comparisons for all prompts."""
    
    print(f"\nRunning comparisons for {len(prompts)} prompts...")
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Comparing responses")):
        # Get responses for this prompt from all methods
        responses = []
        for method_name in method_names:
            if i < len(method_responses[method_name]):
                responses.append(method_responses[method_name][i])
            else:
                responses.append("No response available")
        
        # Compare all responses
        comparison_result = judge.compare_responses(prompt, responses, method_names, persona)
        
        result = {
            "prompt_idx": i,
            "prompt": prompt,
            "responses": dict(zip(method_names, responses)),
            "judgment": comparison_result
        }
        results.append(result)
        
        # Print progress every 10 comparisons
        if (i + 1) % 10 == 0:
            stats = judge.get_statistics()
            print(f"Progress: {i+1}/{len(prompts)}, Cache hit rate: {stats['cache_hits']}/{stats['total_requests']}")
    
    return results


def analyze_results(results: List[Dict[str, Any]], method_names: List[str]) -> Dict[str, Any]:
    """Analyze comparison results and compute statistics."""
    
    # Aggregate scores and rankings
    method_scores = {name: [] for name in method_names}
    method_rankings = {name: [] for name in method_names}
    win_counts = {name: 0 for name in method_names}
    
    for result in results:
        judgment = result["judgment"]
        
        # Collect scores and rankings
        for method_name in method_names:
            if method_name in judgment["scores"]:
                method_scores[method_name].append(judgment["scores"][method_name])
            if method_name in judgment["rankings"]:
                method_rankings[method_name].append(judgment["rankings"][method_name])
        
        # Count wins
        winner = judgment.get("winner", "unknown")
        if winner in win_counts:
            win_counts[winner] += 1
    
    # Compute statistics
    stats = {}
    for method_name in method_names:
        scores = method_scores[method_name]
        rankings = method_rankings[method_name]
        
        stats[method_name] = {
            "mean_score": np.mean(scores) if scores else 0.0,
            "std_score": np.std(scores) if scores else 0.0,
            "mean_ranking": np.mean(rankings) if rankings else 0.0,
            "win_count": win_counts[method_name],
            "win_rate": win_counts[method_name] / len(results) if results else 0.0
        }
    
    return {
        "method_stats": stats,
        "total_comparisons": len(results),
        "method_names": method_names
    }


def print_results_table(analysis: Dict[str, Any]):
    """Print formatted results table."""
    
    method_stats = analysis["method_stats"]
    method_names = analysis["method_names"]
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Table header
    header = f"{'Method':<20} {'Mean Score':<12} {'Mean Rank':<12} {'Win Rate':<12} {'Wins':<8}"
    print(header)
    print("-" * 80)
    
    # Sort methods by mean score (descending)
    sorted_methods = sorted(method_names, key=lambda x: method_stats[x]["mean_score"], reverse=True)
    
    for method_name in sorted_methods:
        stats = method_stats[method_name]
        row = (
            f"{method_name:<20} "
            f"{stats['mean_score']:.2f}±{stats['std_score']:.2f}  "
            f"{stats['mean_ranking']:.2f}        "
            f"{stats['win_rate']:.1%}        "
            f"{stats['win_count']:<8}"
        )
        print(row)
    
    print("="*80)
    print(f"Total comparisons: {analysis['total_comparisons']}")


def save_results(
    results: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    output_path: str,
    args
):
    """Save detailed results to JSON file."""
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method_names": args.method_names,
            "response_files": args.response_files,
            "judge_model": args.judge_model,
            "persona": args.persona,
            "total_comparisons": len(results)
        },
        "analysis": analysis,
        "detailed_results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare pre-generated responses using LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input files
    parser.add_argument(
        "--response_files",
        nargs="+",
        required=True,
        help="Paths to JSON files containing generated responses"
    )
    
    parser.add_argument(
        "--method_names",
        nargs="+",
        required=True,
        help="Names for each method (must match number of response files)"
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
        default="cache/comparison_judge",
        help="Directory to cache judge responses"
    )
    
    parser.add_argument(
        "--persona",
        type=str,
        default="A helpful and harmless assistant",
        help="Persona description for evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/response_comparison.json",
        help="Path to save comparison results"
    )
    
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to compare (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.response_files) != len(args.method_names):
        raise ValueError("Number of response files must match number of method names")
    
    # Check file existence
    for file_path in args.response_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Response file not found: {file_path}")
    
    print("="*80)
    print("RESPONSE COMPARISON EVALUATION")
    print("="*80)
    print(f"Methods: {', '.join(args.method_names)}")
    print(f"Judge: {args.judge_model}")
    print(f"Endpoint: {args.judge_base_url}")
    print(f"Persona: {args.persona}")
    
    # Load and extract responses
    print("\nLoading response files...")
    prompts, method_responses = extract_responses_from_files(args.response_files, args.method_names)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        for method_name in method_responses:
            method_responses[method_name] = method_responses[method_name][:args.max_prompts]
        print(f"Limited to {args.max_prompts} prompts for testing")
    
    # Initialize judge
    print(f"\nInitializing LLM judge...")
    judge = VLLMJudge(
        base_url=args.judge_base_url,
        model=args.judge_model,
        cache_dir=args.judge_cache_dir,
        temperature=0.1,
        max_tokens=1024
    )
    
    # Run comparisons
    results = run_comparisons(prompts, method_responses, args.method_names, judge, args.persona)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results, args.method_names)
    
    # Print results
    print_results_table(analysis)
    
    # Print judge statistics
    stats = judge.get_statistics()
    print(f"\nJudge Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  API calls: {stats['api_calls']}")
    print(f"  Cache hit rate: {stats['cache_hits']/stats['total_requests']:.1%}")
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, analysis, str(output_path), args)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()