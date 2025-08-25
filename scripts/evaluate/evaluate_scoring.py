import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from src.core.drift import get_log_probs

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from vllm import LLM
from transformers import AutoTokenizer

def debug_drift_scoring(data_sample, model, p, base_prompt, attribute_prompts, device, tokenizer, max_examples=3):
    """
    Debug the drift scoring process to understand why selections are similar.
    
    Args:
        data_sample: Small sample of (prompt, output_list) for debugging
        model, p, etc.: Your standard parameters
        max_examples: Number of examples to analyze in detail
    """
    
    print("=== DEBUGGING DRIFT SCORING ===")
    print(f"P-vector: {p}")
    print(f"P-vector norms: L1={torch.norm(torch.tensor(p), p=1):.4f}, L2={torch.norm(torch.tensor(p), p=2):.4f}")
    
    # Analyze a few examples in detail
    for example_idx in range(min(max_examples, len(data_sample))):
        prompt, output_list = data_sample[example_idx]
        n_outputs = len(output_list)
        
        print(f"\n--- EXAMPLE {example_idx + 1} ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Number of outputs: {n_outputs}")
        
        # Get base probabilities for all outputs
        base_probs, base_counts = get_log_probs(
            model, tokenizer, [base_prompt] * n_outputs,
            [prompt] * n_outputs, output_list, device
        )
        base_scores = [p/c for p, c in zip(base_probs, base_counts)]
        
        print(f"Base scores range: [{min(base_scores):.4f}, {max(base_scores):.4f}], std: {np.std(base_scores):.4f}")
        
        # Track individual attribute contributions
        attribute_contributions = torch.zeros(n_outputs, len(attribute_prompts))
        total_drift_scores = torch.zeros(n_outputs)
        
        for attr_idx, attr_prompt in enumerate(attribute_prompts):
            if abs(p[attr_idx]) < 1e-6:
                print(f"Skipping attribute {attr_idx} (p={p[attr_idx]:.6f})")
                continue
                
            # Get attribute probabilities
            attr_probs, attr_counts = get_log_probs(
                model, tokenizer, [attr_prompt] * n_outputs,
                [prompt] * n_outputs, output_list, device
            )
            attr_scores = [p/c for p, c in zip(attr_probs, attr_counts)]
            
            # Compute differences and contributions
            differences = [attr_score - base_score for attr_score, base_score in zip(attr_scores, base_scores)]
            contributions = [p[attr_idx] * diff for diff in differences]
            
            attribute_contributions[:, attr_idx] = torch.tensor(contributions)
            total_drift_scores += torch.tensor(contributions)
            
            print(f"Attr {attr_idx} (p={p[attr_idx]:.3f}): "
                  f"diff_range=[{min(differences):.4f}, {max(differences):.4f}], "
                  f"contrib_range=[{min(contributions):.4f}, {max(contributions):.4f}], "
                  f"contrib_std={np.std(contributions):.4f}")
        
        print(f"Final drift scores range: [{total_drift_scores.min():.4f}, {total_drift_scores.max():.4f}], "
              f"std: {total_drift_scores.std():.4f}")
        
        # Find best and worst outputs
        best_idx = torch.argmax(total_drift_scores).item()
        worst_idx = torch.argmin(total_drift_scores).item()
        
        print(f"Best output (idx {best_idx}, score {total_drift_scores[best_idx]:.4f}): {output_list[best_idx][:100]}...")
        print(f"Worst output (idx {worst_idx}, score {total_drift_scores[worst_idx]:.4f}): {output_list[worst_idx][:100]}...")
        
        # Show attribute breakdown for best output
        print(f"Best output attribute breakdown:")
        for attr_idx in range(len(attribute_prompts)):
            contrib = attribute_contributions[best_idx, attr_idx].item()
            if abs(contrib) > 1e-4:
                print(f"  Attr {attr_idx}: {contrib:.4f} (p={p[attr_idx]:.3f})")
    
    return True

def compare_user_scoring_patterns(data_sample, model, p1, p2, base_prompt, attribute_prompts, device, tokenizer, user1_name="user1", user2_name="user3"):
    """
    Compare how two different p-vectors score the same outputs.
    """
    
    print(f"\n=== COMPARING {user1_name.upper()} vs {user2_name.upper()} SCORING ===")
    
    # Get scores for both users on same data
    scores1_list = []
    scores2_list = []
    
    for prompt, output_list in data_sample[:3]:  # Just first 3 examples
        n_outputs = len(output_list)
        
        # Get base probabilities (same for both users)
        base_probs, base_counts = get_log_probs(
            model, tokenizer, [base_prompt] * n_outputs,
            [prompt] * n_outputs, output_list, device
        )
        base_tensor = torch.tensor([p/c for p, c in zip(base_probs, base_counts)], device=device)
        
        # Compute scores for user 1
        scores1 = torch.zeros(n_outputs, device=device)
        for i, attr_prompt in enumerate(attribute_prompts):
            if abs(p1[i]) < 1e-6:
                continue
            attr_probs, attr_counts = get_log_probs(
                model, tokenizer, [attr_prompt] * n_outputs,
                [prompt] * n_outputs, output_list, device
            )
            attr_tensor = torch.tensor([p/c for p, c in zip(attr_probs, attr_counts)], device=device)
            scores1 += p1[i] * (attr_tensor - base_tensor)
        
        # Compute scores for user 2  
        scores2 = torch.zeros(n_outputs, device=device)
        for i, attr_prompt in enumerate(attribute_prompts):
            if abs(p2[i]) < 1e-6:
                continue
            attr_probs, attr_counts = get_log_probs(
                model, tokenizer, [attr_prompt] * n_outputs,
                [prompt] * n_outputs, output_list, device
            )
            attr_tensor = torch.tensor([p/c for p, c in zip(attr_probs, attr_counts)], device=device)
            scores2 += p2[i] * (attr_tensor - base_tensor)
        
        scores1_list.extend(scores1.cpu().tolist())
        scores2_list.extend(scores2.cpu().tolist())
        
        # Show selections for this prompt
        best1 = torch.argmax(scores1).item()
        best2 = torch.argmax(scores2).item()
        
        print(f"Prompt: {prompt[:50]}...")
        print(f"{user1_name} selects output {best1} (score: {scores1[best1]:.4f})")
        print(f"{user2_name} selects output {best2} (score: {scores2[best2]:.4f})")
        if best1 == best2:
            print("  -> SAME SELECTION!")
        print()
    
    # Overall correlation
    correlation = np.corrcoef(scores1_list, scores2_list)[0, 1]
    print(f"Score correlation between users: {correlation:.4f}")
    
    return scores1_list, scores2_list, correlation

def analyze_attribute_prompt_effects(sample_outputs, model, base_prompt, attribute_prompts, device, tokenizer, sample_prompt="Test prompt"):
    """
    Analyze how much each attribute prompt actually changes the probabilities.
    """
    
    print(f"\n=== ATTRIBUTE PROMPT EFFECT ANALYSIS ===")
    
    # Get base probabilities
    base_probs, base_counts = get_log_probs(
        model, tokenizer, [base_prompt] * len(sample_outputs),
        [sample_prompt] * len(sample_outputs), sample_outputs, device
    )
    base_scores = np.array([p/c for p, c in zip(base_probs, base_counts)])
    
    print(f"Base scores for {len(sample_outputs)} outputs:")
    print(f"  Range: [{base_scores.min():.6f}, {base_scores.max():.6f}]")
    print(f"  Std: {base_scores.std():.6f}")
    
    # Check each attribute prompt
    for i, attr_prompt in enumerate(attribute_prompts):
        attr_probs, attr_counts = get_log_probs(
            model, tokenizer, [attr_prompt] * len(sample_outputs),
            [sample_prompt] * len(sample_outputs), sample_outputs, device
        )
        attr_scores = np.array([p/c for p, c in zip(attr_probs, attr_counts)])
        
        differences = attr_scores - base_scores
        
        print(f"Attribute {i} ({attr_prompt[:50]}...):")
        print(f"  Differences range: [{differences.min():.6f}, {differences.max():.6f}]")
        print(f"  Differences std: {differences.std():.6f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(differences)):.6f}")

# Example usage:
"""
# Debug your scoring process
debug_drift_scoring(your_data_sample, model, user1_p, base_prompt, attribute_prompts, device, tokenizer)

# Compare two users  
compare_user_scoring_patterns(your_data_sample, model, user1_p, user3_p, base_prompt, attribute_prompts, device, tokenizer)

# Analyze if attribute prompts actually change probabilities meaningfully
analyze_attribute_prompt_effects(some_sample_outputs, model, base_prompt, attribute_prompts, device, tokenizer)
"""

def load_bon_data(data_path: str):
    """Load BON data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)

def load_p_vectors(p_path: str, training_sizes: list):
    """Load p vectors from JSONL file for given training sizes."""
    p_vectors = {}
    with open(p_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            p_vectors[entry["n"]] = np.array(entry["p"])
            break
    return p_vectors

def main():
    parser = argparse.ArgumentParser(description="Evaluate and debug drift scoring")
    parser.add_argument("--bon_data", type=str, default="data/bon_all.json", 
                       help="Path to BON data file")
    parser.add_argument("--p_path1", type=str, default="results/user1_p.jsonl",
                       help="Path to first user's p vector file")
    parser.add_argument("--p_path2", type=str, default="results/user3_p.jsonl", 
                       help="Path to second user's p vector file")
    parser.add_argument("--training_sizes", type=str, default="200",
                       help="Comma-separated training sizes to analyze")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Model ID for scoring")
    parser.add_argument("--max_prompts", type=int, default=10,
                       help="Maximum number of prompts to analyze")
    parser.add_argument("--max_outputs", type=int, default=50,
                       help="Maximum number of outputs per prompt to analyze")
    
    args = parser.parse_args()

    from src.core.attribute_prompts import attribute_prompts, base_prompt
    
    # Parse training sizes
    training_sizes = [int(x.strip()) for x in args.training_sizes.split(",")]
    
    print(f"Loading BON data from {args.bon_data}")
    bon_data = load_bon_data(args.bon_data)
    
    # Limit data for analysis
    data_sample = []
    for item in bon_data[:args.max_prompts]:
        prompt = item["prompt"]
        outputs = item["outputs"][:args.max_outputs]
        data_sample.append((prompt, outputs))

    # selected_indices = [0, 1, 2, 31, 33, 37, 43]
    # attribute_prompts = [attribute_prompts[i] for i in selected_indices]
    
    print(f"Analyzing {len(data_sample)} prompts with up to {args.max_outputs} outputs each")
    
    # Setup model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model: {args.model_id}")
    model = LLM(model=args.model_id, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load p vectors
    print(f"Loading p vectors from {args.p_path1}")
    p_vectors1 = load_p_vectors(args.p_path1, training_sizes)
    
    if os.path.exists(args.p_path2):
        print(f"Loading p vectors from {args.p_path2}")
        p_vectors2 = load_p_vectors(args.p_path2, training_sizes)
    else:
        p_vectors2 = None
        print(f"Second p vector file not found: {args.p_path2}")
    
    # Analyze each training size
    for training_size in training_sizes:
        print(f"\n{'='*60}")
        print(f"ANALYZING TRAINING SIZE: {training_size}")
        print(f"{'='*60}")
        
        if training_size not in p_vectors1:
            print(f"No p vector found for training size {training_size} in {args.p_path1}")
            continue
            
        p1 = p_vectors1[training_size]
        
        # Sparsify p vector (keep top 7 elements by absolute value)
        abs_p = np.abs(p1)
        top_indices = np.argsort(abs_p)[-7:]  # Get indices of top 7
        p1_sparse = np.zeros_like(p1)
        p1_sparse[top_indices] = p1[top_indices]
        
        print(f"Original p vector shape: {p1.shape}")
        print(f"Sparsified p vector (top 7): {p1_sparse}")
        print(f"Non-zero elements: {np.count_nonzero(p1_sparse)}")
        
        # Debug drift scoring
        debug_drift_scoring(data_sample, model, p1_sparse, base_prompt, 
                          attribute_prompts, device, tokenizer, max_examples=3)
        
        # Compare with second user if available
        if p_vectors2 and training_size in p_vectors2:
            p2 = p_vectors2[training_size]
            
            # Sparsify second p vector
            abs_p2 = np.abs(p2)
            top_indices2 = np.argsort(abs_p2)[-7:]
            p2_sparse = np.zeros_like(p2)
            p2_sparse[top_indices2] = p2[top_indices2]
            
            user1_name = os.path.basename(args.p_path1).replace("_p.json", "")
            user2_name = os.path.basename(args.p_path2).replace("_p.json", "")
            
            compare_user_scoring_patterns(data_sample, model, p1_sparse, p2_sparse,
                                        base_prompt, attribute_prompts, device, tokenizer,
                                        user1_name, user2_name)
    
    # Analyze attribute prompt effects with sample outputs
    sample_outputs = []
    for prompt, outputs in data_sample[:3]:
        sample_outputs.extend(outputs[:5])  # Get first 5 outputs from first 3 prompts
    
    if sample_outputs:
        analyze_attribute_prompt_effects(sample_outputs, model, base_prompt, 
                                       attribute_prompts, device, tokenizer)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()