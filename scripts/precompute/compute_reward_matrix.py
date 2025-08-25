#!/usr/bin/env python3
"""
Precompute reward matrix Y from persona responses using attribute-based drift scoring.

This script loads persona responses and computes a reward matrix Y where:
- Y[a,p,q] = drift score for persona p's response to question q against attribute a
- Attributes are from attributes/attribute.py (96 different behavioral attributes)  
- Uses the get_scores method from drift.py for consistency with existing codebase
- Results are saved as numpy arrays for use in sparse coding

Matrix format: Y is (num_attributes, num_personas, num_questions)
- Dimension 0: attributes (behavioral styles from attribute.py)
- Dimension 1: personas (different persona prompts)
- Dimension 2: questions/outputs (responses to different questions)

Usage:
    python scripts/precompute/compute_reward_matrix.py \
        --persona-file data/persona_responses.json \
        --output-file data/reward_matrix.npz \
        --scoring-model meta-llama/Llama-3.1-8B-Instruct \
        --reference-prompt "You are a helpful assistant."
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vllm import LLM
from transformers import AutoTokenizer
from src.core.drift import get_log_probs
from attributes.attribute import attribute_prompts


def load_persona_data(persona_file: str, max_personas: int = None, max_questions: int = None) -> Tuple[List[str], List[List[str]], List[str], Dict]:
    """
    Load persona response data from JSON file.
    
    Args:
        persona_file: Path to persona responses JSON file
        max_personas: Maximum number of personas to load
        max_questions: Maximum number of questions per persona
        
    Returns:
        Tuple of (questions, responses_matrix, persona_prompts, metadata)
        where responses_matrix[u][q] = persona u's response to question q
    """
    with open(persona_file, 'r') as f:
        data = json.load(f)
    
    personas = data["personas"]
    if max_personas:
        personas = personas[:max_personas]
    
    # Extract questions from metadata or first persona
    if "questions" in data["metadata"]:
        questions = data["metadata"]["questions"]
    else:
        questions = [r["question"] for r in personas[0]["responses"]]
    
    if max_questions:
        questions = questions[:max_questions]
    
    # Build response matrix: responses[persona_idx][question_idx] = response
    responses_matrix = []
    persona_prompts = []
    
    for persona in personas:
        persona_prompts.append(persona["persona_prompt"])
        persona_responses = [r["response"] for r in persona["responses"]]
        
        if max_questions:
            persona_responses = persona_responses[:max_questions]
        
        responses_matrix.append(persona_responses)
    
    metadata = {
        "num_questions": len(questions),
        "num_personas": len(personas),
        "original_file": persona_file,
        "questions": questions,
        "persona_prompts": persona_prompts
    }
    
    return questions, responses_matrix, persona_prompts, metadata


def compute_individual_attribute_scores(
    model: LLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    responses: List[str],
    reference_prompt: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute individual attribute scores (not summed) for a single persona.
    
    This is a modified version of get_scores that returns individual attribute scores
    instead of the combined drift score.
    
    Args:
        model: vLLM model for scoring
        tokenizer: Model tokenizer
        questions: List of questions
        responses: List of responses (same length as questions)
        reference_prompt: Reference system prompt (base policy)
        device: Device for computation
        
    Returns:
        Array of reward scores (num_questions, num_attributes)
    """
    n = len(questions)
    assert len(responses) == n, "Questions and responses must have same length"
    
    num_attributes = len(attribute_prompts)
    print(f"Computing individual scores for {num_attributes} attributes...")
    
    # Flatten questions and responses for batch processing
    flat_questions = []
    flat_responses = []
    
    for question, response in zip(questions, responses):
        flat_questions.append(question)
        flat_responses.append(response)
    
    total_items = len(flat_responses)
    
    # Get base log probabilities for all items
    print("Computing base log probabilities...")
    base_probs, base_counts = get_log_probs(
        model, tokenizer, [reference_prompt] * total_items, 
        flat_questions, flat_responses, device
    )
    base_tensor = torch.tensor(base_probs, device=device) / torch.tensor(base_counts, device=device)
    
    # Initialize scores matrix: (num_questions, num_attributes)
    scores = torch.zeros((n, num_attributes), device=device)
    
    # Process each attribute individually to get separate scores
    for attr_idx, attribute_prompt in enumerate(tqdm(attribute_prompts, desc="Processing attributes")):
        print(f"Processing attribute {attr_idx+1}/{num_attributes}")
        
        # Get log probabilities for this attribute prompt
        attr_probs, attr_counts = get_log_probs(
            model, tokenizer, [attribute_prompt] * total_items, 
            flat_questions, flat_responses, device
        )
        
        # Convert to tensors
        attr_tensor = torch.tensor(attr_probs, device=device) / torch.tensor(attr_counts, device=device)
        
        # Compute drift for this attribute: attr - base
        attribute_scores = attr_tensor - base_tensor
        
        # Store in scores matrix
        scores[:, attr_idx] = attribute_scores
    
    return scores.cpu().numpy()


def compute_full_reward_matrix(
    persona_file: str,
    scoring_model_name: str,
    reference_prompt: str,
    max_personas: int = None,
    max_questions: int = None,
    device: str = "cuda"
) -> Tuple[np.ndarray, Dict]:
    """
    Compute full reward matrix Y from persona responses using attribute-based drift scoring.
    
    For each persona's responses, computes drift scores against all attributes from
    attributes/attribute.py compared to a reference prompt. Uses get_scores method from drift.py.
    
    Args:
        persona_file: Path to persona responses JSON
        scoring_model_name: Model name for vLLM
        reference_prompt: System prompt for reference policy (base)
        max_personas: Max personas to process
        max_questions: Max questions per persona
        device: Device for computation
        
    Returns:
        Tuple of (Y, metadata) where Y is (num_attributes, num_personas, num_questions)
    """
    print("Loading persona data...")
    questions, responses_matrix, persona_prompts, metadata = load_persona_data(
        persona_file, max_personas, max_questions
    )
    
    num_questions = len(questions)
    num_personas = len(responses_matrix)
    
    print(f"Loaded {num_personas} personas × {num_questions} questions")
    print(f"Total responses to score: {num_personas * num_questions}")
    
    # Initialize model
    print(f"Loading scoring model: {scoring_model_name}")
    model = LLM(
        model=scoring_model_name,
        tensor_parallel_size=1,
        dtype="bfloat16" if device == "cuda" else "float32",
        gpu_memory_utilization=0.5
    )
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
    
    # Get number of attributes
    num_attributes = len(attribute_prompts)
    
    # Initialize reward matrix Y[attribute, persona, question]
    Y = np.zeros((num_attributes, num_personas, num_questions), dtype=np.float32)
    
    print(f"Computing reward scores for {num_attributes} attributes...")
    print(f"Matrix shape: ({num_attributes} attributes, {num_personas} personas, {num_questions} questions)")
    
    # Process each persona
    for persona_idx in tqdm(range(num_personas), desc="Processing personas"):
        persona_responses = responses_matrix[persona_idx]
        
        # Compute individual attribute scores for this persona's responses
        rewards = compute_individual_attribute_scores(
            model, tokenizer,
            questions,
            persona_responses,
            reference_prompt,
            device
        )
        
        # rewards is (num_questions, num_attributes)
        # Transpose and store: Y[attribute, persona, question] = rewards[question, attribute]
        for attr_idx in range(num_attributes):
            for q_idx in range(num_questions):
                Y[attr_idx, persona_idx, q_idx] = rewards[q_idx, attr_idx]
    
    # Update metadata
    metadata.update({
        "scoring_model": scoring_model_name,
        "reference_prompt": reference_prompt,
        "scoring_method": "drift_based_attributes",
        "matrix_shape": Y.shape,
        "num_attributes": num_attributes,
        "num_personas": num_personas,
        "num_questions": num_questions,
        "attribute_prompts": attribute_prompts,
        "dtype": str(Y.dtype),
        "description": "3D tensor: Y[attribute, persona, question] = drift score for persona's response to question against attribute vs reference prompt"
    })
    
    return Y, metadata


def save_reward_matrix(Y: np.ndarray, metadata: Dict, output_file: str):
    """Save reward matrix and metadata to compressed numpy file."""
    
    # Save as compressed numpy archive
    np.savez_compressed(
        output_file,
        Y=Y,
        questions=np.array(metadata["questions"], dtype=object),
        persona_prompts=np.array(metadata["persona_prompts"], dtype=object),
        attribute_prompts=np.array(metadata["attribute_prompts"], dtype=object),
        metadata=np.array([metadata], dtype=object)[0]  # Store as single object
    )
    
    print(f"Saved reward matrix to {output_file}")
    print(f"  Shape: {Y.shape}")
    print(f"  Size: {Path(output_file).stat().st_size / 1e6:.2f} MB")
    print(f"  Mean reward: {np.mean(Y):.4f}")
    print(f"  Std reward: {np.std(Y):.4f}")
    print(f"  Min reward: {np.min(Y):.4f}")
    print(f"  Max reward: {np.max(Y):.4f}")


def load_reward_matrix(file_path: str) -> Tuple[np.ndarray, List[str], List[str], List[str], Dict]:
    """
    Load reward matrix from compressed numpy file.
    
    Returns:
        Tuple of (Y, questions, persona_prompts, attribute_prompts, metadata)
    """
    data = np.load(file_path, allow_pickle=True)
    
    Y = data["Y"]
    questions = data["questions"].tolist()
    persona_prompts = data["persona_prompts"].tolist()
    attribute_prompts = data["attribute_prompts"].tolist()
    metadata = data["metadata"].item()
    
    return Y, questions, persona_prompts, attribute_prompts, metadata


def analyze_reward_matrix(Y: np.ndarray, questions: List[str], persona_prompts: List[str], attribute_prompts: List[str]):
    """Print basic statistics about the reward matrix."""
    
    print("\n" + "="*60)
    print("REWARD MATRIX ANALYSIS")
    print("="*60)
    
    num_attributes, num_personas, num_questions = Y.shape
    
    print(f"Shape: {Y.shape} (attributes × personas × questions)")
    print(f"  Attributes: {num_attributes}")
    print(f"  Personas: {num_personas}")
    print(f"  Questions: {num_questions}")
    print(f"Data type: {Y.dtype}")
    print(f"Memory usage: {Y.nbytes / 1e6:.2f} MB")
    
    print(f"\nGlobal statistics:")
    print(f"  Mean: {np.mean(Y):.4f}")
    print(f"  Std:  {np.std(Y):.4f}")
    print(f"  Min:  {np.min(Y):.4f}")
    print(f"  Max:  {np.max(Y):.4f}")
    
    # Per-attribute statistics (average across personas and questions)
    attr_means = np.mean(Y, axis=(1, 2))  # Average over personas and questions
    attr_stds = np.std(Y, axis=(1, 2))
    
    print(f"\nPer-attribute statistics:")
    print(f"  Mean reward range: [{np.min(attr_means):.4f}, {np.max(attr_means):.4f}]")
    print(f"  Std reward range:  [{np.min(attr_stds):.4f}, {np.max(attr_stds):.4f}]")
    
    # Show top/bottom attributes
    top_attr_idx = np.argmax(attr_means)
    bottom_attr_idx = np.argmin(attr_means)
    
    print(f"\nHighest scoring attribute (idx {top_attr_idx}, mean={attr_means[top_attr_idx]:.4f}):")
    print(f"  {attribute_prompts[top_attr_idx][:80]}...")
    
    print(f"\nLowest scoring attribute (idx {bottom_attr_idx}, mean={attr_means[bottom_attr_idx]:.4f}):")
    print(f"  {attribute_prompts[bottom_attr_idx][:80]}...")
    
    # Per-persona statistics (average across attributes and questions)
    persona_means = np.mean(Y, axis=(0, 2))  # Average over attributes and questions
    print(f"\nPer-persona statistics:")
    print(f"  Mean reward range: [{np.min(persona_means):.4f}, {np.max(persona_means):.4f}]")
    
    # Per-question statistics (average across attributes and personas)
    question_means = np.mean(Y, axis=(0, 1))  # Average over attributes and personas
    print(f"\nPer-question statistics:")
    print(f"  Mean reward range: [{np.min(question_means):.4f}, {np.max(question_means):.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Precompute reward matrix from persona responses")
    parser.add_argument(
        "--persona-file",
        type=str,
        required=True,
        help="Path to persona responses JSON file"
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
        "--max-questions", 
        type=int,
        help="Maximum number of questions per persona"
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
        "--analyze-only",
        action="store_true",
        help="Only analyze existing reward matrix file"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        print(f"Loading reward matrix from {args.output_file}")
        Y, questions, persona_prompts, attribute_prompts, metadata = load_reward_matrix(args.output_file)
        analyze_reward_matrix(Y, questions, persona_prompts, attribute_prompts)
        return
    
    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute reward matrix
    Y, metadata = compute_full_reward_matrix(
        persona_file=args.persona_file,
        scoring_model_name=args.scoring_model,
        reference_prompt=args.reference_prompt,
        max_personas=args.max_personas,
        max_questions=args.max_questions,
        device=args.device
    )
    
    # Save results
    save_reward_matrix(Y, metadata, args.output_file)
    
    # Show analysis
    analyze_reward_matrix(Y, metadata["questions"], metadata["persona_prompts"], metadata["attribute_prompts"])
    
    print(f"\n✅ Reward matrix computation complete!")
    print(f"Load with: Y, questions, persona_prompts, attribute_prompts, metadata = load_reward_matrix('{args.output_file}')")


if __name__ == "__main__":
    main()