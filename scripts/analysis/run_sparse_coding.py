#!/usr/bin/env python3
"""
Run sparse coding experiments for persona selection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm import LLM
from transformers import AutoTokenizer
from src.core.sparse_coding import SparseCoding
from src.core.reward_matrix import RewardMatrixBuilder
from attributes.personas import persona_prompts


def run_experiment(
    Y: torch.Tensor,
    k: int,
    lambda1: float,
    lambda21: float,
    beta: float,
    epsilon: float,
    max_iter: int = 500,
    device: str = "cuda"
) -> Dict:
    """Run a single sparse coding experiment."""
    
    model = SparseCoding(
        k=k,
        lambda1=lambda1,
        lambda21=lambda21,
        beta=beta,
        epsilon=epsilon,
        max_iter=max_iter,
        device=device
    )
    
    results = model.fit(Y, verbose=True)
    
    # Add selection information
    selected_personas = model.get_selected_personas(threshold=0.1)
    results["selected_personas"] = selected_personas.cpu().numpy().tolist()
    results["num_selected"] = len(selected_personas)
    
    # Compute selection weights for each persona
    persona_weights = torch.sum(torch.abs(model.W), dim=0).cpu().numpy()
    results["persona_weights"] = persona_weights.tolist()
    
    return results


def parameter_sweep(
    Y: torch.Tensor,
    param_grid: Dict[str, List],
    output_dir: Path,
    device: str = "cuda"
) -> List[Dict]:
    """Run experiments with parameter sweep."""
    
    experiments = []
    
    # Generate all parameter combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        
        print(f"\nRunning experiment with params: {params}")
        
        # Run experiment
        results = run_experiment(
            Y,
            k=params["k"],
            lambda1=params["lambda1"],
            lambda21=params["lambda21"],
            beta=params.get("beta", 0.01),
            epsilon=params.get("epsilon", 1e-4),
            device=device
        )
        
        # Add parameters to results
        results["parameters"] = params
        experiments.append(results)
        
        # Save intermediate results
        exp_name = f"k{params['k']}_l1{params['lambda1']}_l21{params['lambda21']}"
        save_path = output_dir / f"{exp_name}.pt"
        torch.save(results, save_path)
        print(f"Saved to {save_path}")
    
    return experiments


def visualize_results(experiments: List[Dict], output_dir: Path):
    """Create visualizations of experimental results."""
    
    # 1. Reconstruction error vs sparsity tradeoff
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Final error vs sparsity
    ax = axes[0, 0]
    errors = [exp["final_error"] for exp in experiments]
    sparsities = [exp["final_sparsity"] for exp in experiments]
    k_values = [exp["parameters"]["k"] for exp in experiments]
    
    scatter = ax.scatter(sparsities, errors, c=k_values, cmap='viridis', s=100)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Error vs Sparsity Tradeoff")
    plt.colorbar(scatter, ax=ax, label="k (basis size)")
    
    # Plot 2: Number of selected personas vs k
    ax = axes[0, 1]
    k_vals = sorted(set(k_values))
    for k in k_vals:
        k_exps = [exp for exp in experiments if exp["parameters"]["k"] == k]
        num_selected = [exp["num_selected"] for exp in k_exps]
        lambda1_vals = [exp["parameters"]["lambda1"] for exp in k_exps]
        ax.scatter(lambda1_vals, num_selected, label=f"k={k}", alpha=0.7)
    
    ax.set_xlabel("λ₁ (L1 regularization)")
    ax.set_ylabel("Number of Selected Personas")
    ax.set_title("Persona Selection vs Regularization")
    ax.legend()
    ax.set_xscale('log')
    
    # Plot 3: Convergence curves
    ax = axes[1, 0]
    for i, exp in enumerate(experiments[:5]):  # Show first 5
        history = exp["history"]
        params = exp["parameters"]
        label = f"k={params['k']}, λ₁={params['lambda1']:.3f}"
        ax.plot(history["reconstruction_error"], label=label, alpha=0.7)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Convergence Curves")
    ax.legend(fontsize=8)
    
    # Plot 4: Active atoms over iterations
    ax = axes[1, 1]
    for i, exp in enumerate(experiments[:5]):  # Show first 5
        history = exp["history"]
        params = exp["parameters"]
        label = f"k={params['k']}, λ₂₁={params['lambda21']:.3f}"
        ax.plot(history["num_active_atoms"], label=label, alpha=0.7)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Active Atoms")
    ax.set_title("Active Atoms During Training")
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sparse_coding_results.png", dpi=150)
    plt.close()
    
    # 2. Persona weight distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Select best experiment (lowest error with good sparsity)
    best_exp = min(experiments, 
                   key=lambda x: x["final_error"] + 10 * (1 - x["final_sparsity"]))
    
    weights = best_exp["persona_weights"]
    selected = best_exp["selected_personas"]
    
    x = np.arange(len(weights))
    colors = ['red' if i in selected else 'blue' for i in range(len(weights))]
    
    ax.bar(x, weights, color=colors, alpha=0.7)
    ax.set_xlabel("Persona Index")
    ax.set_ylabel("Total Weight")
    ax.set_title(f"Persona Weights (k={best_exp['parameters']['k']}, "
                 f"λ₁={best_exp['parameters']['lambda1']}, "
                 f"λ₂₁={best_exp['parameters']['lambda21']})")
    ax.axhline(y=0.1, color='r', linestyle='--', label='Selection threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "persona_weights.png", dpi=150)
    plt.close()


def analyze_selected_personas(
    experiments: List[Dict],
    persona_prompts: List[str],
    output_dir: Path
):
    """Analyze which personas are consistently selected."""
    
    # Count selection frequency across experiments
    persona_counts = np.zeros(len(persona_prompts))
    
    for exp in experiments:
        selected = exp["selected_personas"]
        for idx in selected:
            if idx < len(persona_prompts):
                persona_counts[idx] += 1
    
    # Sort by frequency
    sorted_indices = np.argsort(persona_counts)[::-1]
    
    # Save analysis
    analysis = {
        "selection_frequency": {},
        "top_selected": [],
        "never_selected": []
    }
    
    for idx in sorted_indices:
        count = int(persona_counts[idx])
        if count > 0:
            # Extract profession from prompt
            prompt = persona_prompts[idx]
            # Simple extraction - look for "like a X"
            if "like a " in prompt:
                start = prompt.index("like a ") + 7
                end = prompt.index(",", start) if "," in prompt[start:] else prompt.index(".", start)
                profession = prompt[start:end]
            else:
                profession = f"Persona {idx}"
            
            analysis["selection_frequency"][profession] = count
            
            if count > len(experiments) * 0.5:  # Selected in >50% of experiments
                analysis["top_selected"].append({
                    "index": int(idx),
                    "profession": profession,
                    "frequency": count / len(experiments)
                })
        else:
            analysis["never_selected"].append(int(idx))
    
    # Save analysis
    with open(output_dir / "persona_selection_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*60)
    print("TOP SELECTED PERSONAS")
    print("="*60)
    for item in analysis["top_selected"][:10]:
        print(f"{item['profession']}: {item['frequency']:.2%} of experiments")


def main():
    parser = argparse.ArgumentParser(description="Run sparse coding experiments")
    parser.add_argument(
        "--reward-matrix",
        type=str,
        help="Path to saved reward matrix (if already computed)"
    )
    parser.add_argument(
        "--persona-dir",
        type=str,
        default="data/persona_responses",
        help="Directory with persona responses"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sparse_coding",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model for reward computation"
    )
    parser.add_argument(
        "--base-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Base prompt for drift computation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or compute reward matrix
    if args.reward_matrix:
        print(f"Loading reward matrix from {args.reward_matrix}")
        Y, metadata = RewardMatrixBuilder.load_reward_matrix(args.reward_matrix)
        Y = Y.to(args.device)
    else:
        print("Computing reward matrix from persona responses...")
        
        # Initialize model and tokenizer
        model = LLM(
            model=args.model_name,
            tensor_parallel_size=1,
            dtype="bfloat16" if args.device == "cuda" else "float32"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Build reward matrix
        builder = RewardMatrixBuilder(
            model=model,
            tokenizer=tokenizer,
            base_prompt=args.base_prompt,
            attribute_prompts=persona_prompts,
            device=args.device
        )
        
        Y, metadata = builder.compute_reward_matrix(
            args.persona_dir,
            num_personas=100,
            num_questions=100
        )
        
        # Save for reuse
        save_path = output_dir / "reward_matrix.pt"
        builder.save_reward_matrix(Y, metadata, str(save_path))
    
    print(f"Reward matrix shape: {Y.shape}")
    
    if args.sweep:
        # Parameter sweep
        param_grid = {
            "k": [10, 20, 30, 50],
            "lambda1": [0.01, 0.1, 0.5, 1.0],
            "lambda21": [0.01, 0.1, 0.5],
            "beta": [0.01],
            "epsilon": [1e-4]
        }
        
        experiments = parameter_sweep(Y, param_grid, output_dir, args.device)
        
        # Save all experiments
        with open(output_dir / "all_experiments.json", "w") as f:
            # Convert tensors to lists for JSON serialization
            json_experiments = []
            for exp in experiments:
                json_exp = {
                    "parameters": exp["parameters"],
                    "final_error": exp["final_error"],
                    "final_sparsity": exp["final_sparsity"],
                    "num_selected": exp["num_selected"],
                    "selected_personas": exp["selected_personas"]
                }
                json_experiments.append(json_exp)
            json.dump(json_experiments, f, indent=2)
        
        # Visualize results
        visualize_results(experiments, output_dir)
        
        # Analyze selected personas
        analyze_selected_personas(experiments, persona_prompts, output_dir)
        
    else:
        # Single experiment with default parameters
        results = run_experiment(
            Y,
            k=30,
            lambda1=0.1,
            lambda21=0.1,
            beta=0.01,
            epsilon=1e-4,
            device=args.device
        )
        
        print(f"\nResults:")
        print(f"  Final error: {results['final_error']:.4f}")
        print(f"  Final sparsity: {results['final_sparsity']:.3f}")
        print(f"  Selected personas: {results['num_selected']}/{Y.shape[1]}")
        
        # Save results
        torch.save(results, output_dir / "single_experiment.pt")
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()