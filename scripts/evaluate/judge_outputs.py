#!/usr/bin/env python3
"""
Judge generated outputs using persona LLM judge.
Loads outputs from generation scripts and evaluates them.
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.judges.llm_judge import PersonaJudge


def load_generated_outputs(input_path: str) -> List[Dict]:
    """Load generated outputs from file."""
    with open(input_path, 'r') as f:
        return json.load(f)


async def judge_single_method(outputs: List[Dict], judge: PersonaJudge, method_name: str) -> List[Dict]:
    """Judge outputs from a single generation method."""
    results = []
    
    for item in tqdm(outputs, desc=f"Judging {method_name}"):
        prompt = item['prompt']
        persona = item.get('persona', 'A helpful assistant')
        candidates = item['outputs']
        
        if len(candidates) <= 1:
            # Only one output or none, just record it
            results.append({
                "prompt": prompt,
                "persona": persona,
                "method": method_name,
                "best_output": candidates[0] if candidates else "",
                "best_index": 0,
                "total_outputs": len(candidates),
                "comparisons": 0
            })
            continue
        
        # Tournament selection using pairwise comparisons
        best_output = candidates[0]
        best_index = 0
        comparisons = 0
        
        # Compare current best against all others
        for i, candidate in enumerate(candidates[1:], 1):
            winner = await judge.compare_responses(persona, prompt, best_output, candidate)
            comparisons += 1
            
            if winner == "B":  # Candidate wins
                best_output = candidate
                best_index = i
        
        results.append({
            "prompt": prompt,
            "persona": persona,
            "method": method_name,
            "best_output": best_output,
            "best_index": best_index,
            "total_outputs": len(candidates),
            "comparisons": comparisons
        })
    
    return results


async def compare_methods(method_results: Dict[str, List[Dict]], judge: PersonaJudge) -> List[Dict]:
    """Compare best outputs across different methods."""
    # Group by prompt
    by_prompt = {}
    for method, results in method_results.items():
        for result in results:
            prompt = result['prompt']
            if prompt not in by_prompt:
                by_prompt[prompt] = {}
            by_prompt[prompt][method] = result
    
    comparisons = []
    
    for prompt, method_outputs in tqdm(by_prompt.items(), desc="Comparing methods"):
        methods = list(method_outputs.keys())
        
        if len(methods) < 2:
            continue
        
        persona = list(method_outputs.values())[0]['persona']
        
        # Compare all pairs of methods
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method_a, method_b = methods[i], methods[j]
                output_a = method_outputs[method_a]['best_output']
                output_b = method_outputs[method_b]['best_output']
                
                winner = await judge.compare_responses(persona, prompt, output_a, output_b)
                
                comparisons.append({
                    "prompt": prompt,
                    "persona": persona,
                    "method_a": method_a,
                    "method_b": method_b,
                    "output_a": output_a,
                    "output_b": output_b,
                    "winner": "A" if winner == "A" else "B",
                    "winning_method": method_a if winner == "A" else method_b
                })
    
    return comparisons


def print_statistics(method_results: Dict[str, List[Dict]], comparisons: List[Dict] = None):
    """Print judging statistics."""
    print("\n" + "="*60)
    print("JUDGING STATISTICS")
    print("="*60)
    
    # Individual method stats
    for method, results in method_results.items():
        total = len(results)
        selection_dist = {}
        
        for result in results:
            idx = result['best_index']
            selection_dist[idx] = selection_dist.get(idx, 0) + 1
        
        print(f"\n{method}:")
        print(f"  Total prompts: {total}")
        print(f"  Selection distribution:")
        for idx in sorted(selection_dist.keys()):
            count = selection_dist[idx]
            pct = 100 * count / total
            print(f"    Position {idx}: {count}/{total} ({pct:.1f}%)")
    
    # Method comparison stats
    if comparisons:
        print(f"\nMethod Comparisons:")
        method_wins = {}
        for comp in comparisons:
            winner = comp['winning_method']
            method_wins[winner] = method_wins.get(winner, 0) + 1
        
        total_comps = len(comparisons)
        for method, wins in sorted(method_wins.items()):
            pct = 100 * wins / total_comps
            print(f"  {method}: {wins}/{total_comps} ({pct:.1f}%)")


async def main():
    parser = argparse.ArgumentParser(description="Judge generated outputs")
    
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files with generated outputs")
    parser.add_argument("--output", type=str, default="judging_results.json", help="Output file")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Judge model")
    parser.add_argument("--judge_url", type=str, default="http://localhost:8000/v1", help="Judge API URL")
    parser.add_argument("--compare_methods", action="store_true", help="Compare best outputs across methods")
    parser.add_argument("--max_prompts", type=int, default=None, help="Max prompts per method")
    
    args = parser.parse_args()
    
    # Initialize judge
    print("Initializing judge...")
    judge = PersonaJudge(base_url=args.judge_url, model=args.judge_model)
    
    # Load and judge each method
    method_results = {}
    
    for input_file in args.inputs:
        print(f"\nLoading outputs from: {input_file}")
        outputs = load_generated_outputs(input_file)
        
        if args.max_prompts:
            outputs = outputs[:args.max_prompts]
        
        # Get method name from outputs or filename
        method_name = outputs[0].get('method', Path(input_file).stem) if outputs else Path(input_file).stem
        
        print(f"Judging {len(outputs)} samples for method: {method_name}")
        results = await judge_single_method(outputs, judge, method_name)
        method_results[method_name] = results
    
    # Compare methods if requested
    comparisons = []
    if args.compare_methods and len(method_results) > 1:
        print("\nComparing methods...")
        comparisons = await compare_methods(method_results, judge)
    
    # Save results
    output_data = {
        "method_results": method_results,
        "comparisons": comparisons,
        "metadata": {
            "judge_model": args.judge_model,
            "total_methods": len(method_results),
            "total_comparisons": len(comparisons)
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print statistics
    print_statistics(method_results, comparisons)


if __name__ == "__main__":
    asyncio.run(main())