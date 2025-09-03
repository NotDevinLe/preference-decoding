#!/usr/bin/env python3
"""
Compute reward matrix from persona preference data with flexible question handling.

This script loads persona preference data and computes a flattened reward matrix where:
- Each row represents a single (persona, question-response) pair
- Each column represents an attribute score
- Does NOT assume all personas have the same questions
- Processes all *_train.json files in data/persona_pref/

The output format maintains compatibility with the original compute_reward_matrix.py
but uses a flattened structure since questions vary across personas.

Usage:
    python scripts/precompute/compute_reward_matrix_flexible.py \
        --data-dir data/persona_pref \
        --output-file data/reward_matrix_flexible.npz \
        --scoring-model meta-llama/Llama-3.1-8B-Instruct \
        --reference-prompt "You are a helpful assistant."
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
from glob import glob

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm import LLM
from transformers import AutoTokenizer
from src.core.drift import get_log_probs
from attributes.attribute import attribute_prompts


def load_persona_preference_data(file_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str], List[str], str]:
    """
    Load persona preference data from a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (prompts, chosen_responses, rejected_responses, persona_id)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    prompts = [item["prompt"] for item in data]
    chosen = [item["chosen"] for item in data]
    rejected = [item["rejected"] for item in data]
    
    # Extract persona ID from filename (e.g., user_11_train.json -> user_11)
    persona_id = Path(file_path).stem.replace("_train", "")
    
    return prompts, chosen, rejected, persona_id


def load_all_training_data(data_dir: str, max_personas: Optional[int] = None, 
                          max_samples_per_persona: Optional[int] = None) -> Dict:
    """
    Load all training data from the persona_pref directory.
    
    Args:
        data_dir: Directory containing *_train.json files
        max_personas: Maximum number of personas to load
        max_samples_per_persona: Maximum samples per persona
        
    Returns:
        Dictionary with loaded data organized by persona
    """
    train_files = sorted(glob(f"{data_dir}/*_train.json"))
    
    if max_personas:
        train_files = train_files[:max_personas]
    
    print(f"Found {len(train_files)} training files")
    
    all_data = {
        "personas": [],
        "all_prompts": [],
        "all_chosen": [],
        "all_rejected": [],
        "persona_indices": [],  # Track which persona each sample belongs to
        "prompt_indices": [],   # Track prompt index within each persona
    }
    
    for file_idx, file_path in enumerate(train_files):
        prompts, chosen, rejected, persona_id = load_persona_preference_data(
            file_path, max_samples_per_persona
        )
        
        all_data["personas"].append(persona_id)
        
        # Add all samples with tracking indices
        for prompt_idx, (p, c, r) in enumerate(zip(prompts, chosen, rejected)):
            all_data["all_prompts"].append(p)
            all_data["all_chosen"].append(c)
            all_data["all_rejected"].append(r)
            all_data["persona_indices"].append(file_idx)
            all_data["prompt_indices"].append(prompt_idx)
        
        print(f"Loaded {len(prompts)} samples from {persona_id}")
    
    all_data["total_samples"] = len(all_data["all_prompts"])
    all_data["num_personas"] = len(all_data["personas"])
    
    return all_data


def compute_attribute_scores_batch(
    model: LLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    responses: List[str],
    reference_prompt: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute individual attribute scores for a batch of question-response pairs.
    
    Args:
        model: vLLM model for scoring
        tokenizer: Model tokenizer
        questions: List of questions
        responses: List of responses
        reference_prompt: Reference system prompt
        device: Device for computation
        
    Returns:
        Array of scores (num_samples, num_attributes)
    """
    n = len(questions)
    assert len(responses) == n, "Questions and responses must have same length"
    
    num_attributes = len(attribute_prompts)
    
    # Get base log probabilities
    base_probs, base_counts = get_log_probs(
        model, tokenizer, [reference_prompt] * n, 
        questions, responses, device
    )
    base_tensor = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Initialize scores matrix
    scores = torch.zeros((n, num_attributes), device=device)
    
    # Process each attribute
    for attr_idx, attribute_prompt in enumerate(tqdm(attribute_prompts, desc="Processing attributes", leave=False)):
        # Get log probabilities for this attribute
        attr_probs, attr_counts = get_log_probs(
            model, tokenizer, [attribute_prompt] * n, 
            questions, responses, device
        )
        
        attr_tensor = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Compute drift scores
        scores[:, attr_idx] = attr_tensor - base_tensor
    
    return scores.cpu().numpy()


def compute_flexible_reward_matrix(
    data_dir: str,
    scoring_model_name: str,
    reference_prompt: str,
    max_personas: Optional[int] = None,
    max_samples_per_persona: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    process_rejected: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute reward matrices for chosen and rejected responses with flexible question handling.
    
    Args:
        data_dir: Directory containing training files
        scoring_model_name: Model name for vLLM
        reference_prompt: Reference system prompt
        max_personas: Max personas to process
        max_samples_per_persona: Max samples per persona
        batch_size: Batch size for processing
        device: Device for computation
        process_rejected: Whether to also process rejected responses
        
    Returns:
        Tuple of (Y_chosen, Y_rejected, metadata)
        Y_chosen/Y_rejected are 2D: (num_total_samples, num_attributes)
    """
    print("Loading all training data...")
    data = load_all_training_data(data_dir, max_personas, max_samples_per_persona)
    
    total_samples = data["total_samples"]
    num_personas = data["num_personas"]
    
    print(f"Loaded {total_samples} total samples from {num_personas} personas")
    
    # Initialize model
    print(f"Loading scoring model: {scoring_model_name}")
    model = LLM(
        model=scoring_model_name,
        tensor_parallel_size=1,
        dtype="bfloat16" if device == "cuda" else "float32",
        gpu_memory_utilization=0.5
    )
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
    
    num_attributes = len(attribute_prompts)
    
    # Initialize reward matrices - 2D since questions vary
    Y_chosen = np.zeros((total_samples, num_attributes), dtype=np.float32)
    Y_rejected = np.zeros((total_samples, num_attributes), dtype=np.float32) if process_rejected else None
    
    print(f"Computing reward scores for {num_attributes} attributes...")
    print(f"Matrix shape: ({total_samples} samples, {num_attributes} attributes)")
    
    # Process in batches
    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        
        batch_questions = data["all_prompts"][start_idx:end_idx]
        batch_chosen = data["all_chosen"][start_idx:end_idx]
        
        # Compute scores for chosen responses
        chosen_scores = compute_attribute_scores_batch(
            model, tokenizer,
            batch_questions,
            batch_chosen,
            reference_prompt,
            device
        )
        Y_chosen[start_idx:end_idx] = chosen_scores
        
        # Compute scores for rejected responses if requested
        if process_rejected:
            batch_rejected = data["all_rejected"][start_idx:end_idx]
            rejected_scores = compute_attribute_scores_batch(
                model, tokenizer,
                batch_questions,
                batch_rejected,
                reference_prompt,
                device
            )
            Y_rejected[start_idx:end_idx] = rejected_scores
    
    # Create metadata
    metadata = {
        "scoring_model": scoring_model_name,
        "reference_prompt": reference_prompt,
        "scoring_method": "drift_based_attributes",
        "num_attributes": num_attributes,
        "num_personas": num_personas,
        "total_samples": total_samples,
        "personas": data["personas"],
        "persona_indices": data["persona_indices"],
        "prompt_indices": data["prompt_indices"],
        "prompts": data["all_prompts"],
        "chosen_responses": data["all_chosen"],
        "rejected_responses": data["all_rejected"] if process_rejected else None,
        "attribute_prompts": attribute_prompts,
        "Y_chosen_shape": Y_chosen.shape,
        "Y_rejected_shape": Y_rejected.shape if process_rejected else None,
        "dtype": str(Y_chosen.dtype),
        "description": "2D matrices: Y[sample, attribute] = drift score for response against attribute vs reference prompt"
    }
    
    return Y_chosen, Y_rejected, metadata


def save_flexible_reward_matrix(Y_chosen: np.ndarray, Y_rejected: Optional[np.ndarray], 
                                metadata: Dict, output_file: str):
    """Save reward matrices and metadata to compressed numpy file."""
    
    save_dict = {
        "Y": Y_chosen,  # Keep name for compatibility
        "Y_chosen": Y_chosen,
        "persona_indices": np.array(metadata["persona_indices"]),
        "prompt_indices": np.array(metadata["prompt_indices"]),
        "prompts": np.array(metadata["prompts"], dtype=object),
        "chosen_responses": np.array(metadata["chosen_responses"], dtype=object),
        "personas": np.array(metadata["personas"], dtype=object),
        "attribute_prompts": np.array(metadata["attribute_prompts"], dtype=object),
        "metadata": np.array([metadata], dtype=object)[0]
    }
    
    if Y_rejected is not None:
        save_dict["Y_rejected"] = Y_rejected
        save_dict["rejected_responses"] = np.array(metadata["rejected_responses"], dtype=object)
    
    # Save as compressed numpy archive
    np.savez_compressed(output_file, **save_dict)
    
    print(f"\nSaved reward matrix to {output_file}")
    print(f"  Chosen shape: {Y_chosen.shape}")
    if Y_rejected is not None:
        print(f"  Rejected shape: {Y_rejected.shape}")
    print(f"  Size: {Path(output_file).stat().st_size / 1e6:.2f} MB")
    print(f"\nChosen statistics:")
    print(f"  Mean: {np.mean(Y_chosen):.4f}")
    print(f"  Std:  {np.std(Y_chosen):.4f}")
    print(f"  Min:  {np.min(Y_chosen):.4f}")
    print(f"  Max:  {np.max(Y_chosen):.4f}")
    
    if Y_rejected is not None:
        print(f"\nRejected statistics:")
        print(f"  Mean: {np.mean(Y_rejected):.4f}")
        print(f"  Std:  {np.std(Y_rejected):.4f}")
        print(f"  Min:  {np.min(Y_rejected):.4f}")
        print(f"  Max:  {np.max(Y_rejected):.4f}")


def load_flexible_reward_matrix(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Load flexible reward matrix from file.
    
    Returns:
        Tuple of (Y_chosen, Y_rejected, metadata)
    """
    data = np.load(file_path, allow_pickle=True)
    
    Y_chosen = data["Y_chosen"] if "Y_chosen" in data else data["Y"]
    Y_rejected = data["Y_rejected"] if "Y_rejected" in data else None
    metadata = data["metadata"].item()
    
    return Y_chosen, Y_rejected, metadata


def analyze_flexible_reward_matrix(Y_chosen: np.ndarray, Y_rejected: Optional[np.ndarray], metadata: Dict):
    """Print analysis of the flexible reward matrix."""
    
    print("\n" + "="*60)
    print("FLEXIBLE REWARD MATRIX ANALYSIS")
    print("="*60)
    
    num_samples, num_attributes = Y_chosen.shape
    
    print(f"\nData shape:")
    print(f"  Total samples: {num_samples}")
    print(f"  Attributes: {num_attributes}")
    print(f"  Personas: {metadata['num_personas']}")
    print(f"  Memory usage: {Y_chosen.nbytes / 1e6:.2f} MB")
    
    # Samples per persona
    persona_counts = {}
    for p_idx in metadata["persona_indices"]:
        persona_id = metadata["personas"][p_idx]
        persona_counts[persona_id] = persona_counts.get(persona_id, 0) + 1
    
    print(f"\nSamples per persona:")
    for persona_id, count in sorted(persona_counts.items()):
        print(f"  {persona_id}: {count} samples")
    
    print(f"\nChosen response statistics:")
    print(f"  Mean: {np.mean(Y_chosen):.4f}")
    print(f"  Std:  {np.std(Y_chosen):.4f}")
    print(f"  Min:  {np.min(Y_chosen):.4f}")
    print(f"  Max:  {np.max(Y_chosen):.4f}")
    
    if Y_rejected is not None:
        print(f"\nRejected response statistics:")
        print(f"  Mean: {np.mean(Y_rejected):.4f}")
        print(f"  Std:  {np.std(Y_rejected):.4f}")
        print(f"  Min:  {np.min(Y_rejected):.4f}")
        print(f"  Max:  {np.max(Y_rejected):.4f}")
        
        # Preference margin statistics
        margin = Y_chosen - Y_rejected
        print(f"\nPreference margin (chosen - rejected):")
        print(f"  Mean: {np.mean(margin):.4f}")
        print(f"  Std:  {np.std(margin):.4f}")
        print(f"  Min:  {np.min(margin):.4f}")
        print(f"  Max:  {np.max(margin):.4f}")
    
    # Per-attribute statistics
    attr_means_chosen = np.mean(Y_chosen, axis=0)
    attr_stds_chosen = np.std(Y_chosen, axis=0)
    
    print(f"\nPer-attribute statistics (chosen):")
    print(f"  Mean range: [{np.min(attr_means_chosen):.4f}, {np.max(attr_means_chosen):.4f}]")
    print(f"  Std range:  [{np.min(attr_stds_chosen):.4f}, {np.max(attr_stds_chosen):.4f}]")
    
    # Top attributes
    top_idx = np.argmax(attr_means_chosen)
    bottom_idx = np.argmin(attr_means_chosen)
    
    print(f"\nHighest scoring attribute (idx {top_idx}, mean={attr_means_chosen[top_idx]:.4f}):")
    print(f"  {metadata['attribute_prompts'][top_idx][:80]}...")
    
    print(f"\nLowest scoring attribute (idx {bottom_idx}, mean={attr_means_chosen[bottom_idx]:.4f}):")
    print(f"  {metadata['attribute_prompts'][bottom_idx][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Compute flexible reward matrix from persona preference data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/persona_pref",
        help="Directory containing *_train.json files"
    )
    parser.add_argument(
        "--output-file", 
        type=str,
        required=True,
        help="Output path for reward matrix (.npz file)"
    )
    parser.add_argument(
        "--scoring-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for computing log probabilities"
    )
    parser.add_argument(
        "--reference-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Reference prompt for computing log-likelihood ratio"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        help="Maximum number of personas to process"
    )
    parser.add_argument(
        "--max-samples-per-persona",
        type=int,
        help="Maximum number of samples per persona"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--no-rejected",
        action="store_true",
        help="Skip processing rejected responses"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing reward matrix file"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print(f"Loading reward matrix from {args.output_file}")
        Y_chosen, Y_rejected, metadata = load_flexible_reward_matrix(args.output_file)
        analyze_flexible_reward_matrix(Y_chosen, Y_rejected, metadata)
        return
    
    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute reward matrices
    Y_chosen, Y_rejected, metadata = compute_flexible_reward_matrix(
        data_dir=args.data_dir,
        scoring_model_name=args.scoring_model,
        reference_prompt=args.reference_prompt,
        max_personas=args.max_personas,
        max_samples_per_persona=args.max_samples_per_persona,
        batch_size=args.batch_size,
        device=args.device,
        process_rejected=not args.no_rejected
    )
    
    # Save results
    save_flexible_reward_matrix(Y_chosen, Y_rejected, metadata, args.output_file)
    
    # Show analysis
    analyze_flexible_reward_matrix(Y_chosen, Y_rejected, metadata)
    
    print(f"\n✅ Flexible reward matrix computation complete!")
    print(f"Load with: Y_chosen, Y_rejected, metadata = load_flexible_reward_matrix('{args.output_file}')")


if __name__ == "__main__":
    main()