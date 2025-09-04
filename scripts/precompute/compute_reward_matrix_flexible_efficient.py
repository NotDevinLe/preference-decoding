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


def load_sampled_personas(personas_file: str) -> List[str]:
    """
    Load persona prompts from sampled_personas.json
    
    Args:
        personas_file: Path to sampled_personas.json
        
    Returns:
        List of persona prompts
    """
    with open(personas_file, 'r') as f:
        data = json.load(f)
    
    return [p["persona"] for p in data["personas"]]


def load_user_preference_data(file_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str], List[str], str]:
    """
    Load user preference data from a single JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (prompts, chosen_responses, rejected_responses, user_id)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    prompts = [item["prompt"] for item in data]
    chosen = [item["chosen"] for item in data]
    rejected = [item["rejected"] for item in data]
    
    # Extract user ID from filename (e.g., user_11_train.json -> user_11)
    user_id = Path(file_path).stem.replace("_train", "")
    
    return prompts, chosen, rejected, user_id


def load_all_training_data(data_dir: str, max_users: Optional[int] = None, 
                          max_samples_per_user: Optional[int] = None) -> Dict:
    """
    Load all training data from the persona_pref directory.
    
    Args:
        data_dir: Directory containing *_train.json files
        max_users: Maximum number of users to load
        max_samples_per_user: Maximum samples per user
        
    Returns:
        Dictionary with loaded data organized by user
    """
    train_files = sorted(glob(f"{data_dir}/*_train.json"))
    
    if max_users:
        train_files = train_files[:max_users]
    
    print(f"Found {len(train_files)} training files")
    
    all_data = {
        "users": [],
        "user_data": []  # List of dicts, one per user
    }
    
    for file_path in train_files:
        prompts, chosen, rejected, user_id = load_user_preference_data(
            file_path, max_samples_per_user
        )
        
        user_data = {
            "user_id": user_id,
            "prompts": prompts,
            "chosen": chosen,
            "rejected": rejected,
            "num_samples": len(prompts)
        }
        
        all_data["users"].append(user_id)
        all_data["user_data"].append(user_data)
        
        print(f"Loaded {len(prompts)} samples from {user_id}")
    
    all_data["num_users"] = len(all_data["users"])
    
    return all_data


def compute_flexible_reward_matrix_efficient(
    data_dir: str,
    personas_file: str,
    scoring_model_name: str,
    reference_prompt: str,
    max_users: Optional[int] = None,
    max_samples_per_user: Optional[int] = None,
    max_personas: Optional[int] = None,
    device: str = "cuda",
    process_rejected: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Efficiently compute reward matrix with proper batching.
    
    Key improvements:
    1. Collect ALL samples from ALL users first
    2. Compute baseline once for all samples
    3. Process each persona against all samples in one batch
    4. Much better GPU utilization
    """
    
    print("Loading persona prompts...")
    persona_prompts = load_sampled_personas(personas_file)
    if max_personas:
        persona_prompts = persona_prompts[:max_personas]
    
    print(f"Using {len(persona_prompts)} personas as attributes")
    
    print("Loading user training data...")
    data = load_all_training_data(data_dir, max_users, max_samples_per_user)
    
    num_users = data["num_users"]
    num_personas = len(persona_prompts)
    
    print(f"Processing {num_users} users with {num_personas} personas")
    
    # STEP 1: Collect ALL samples from ALL users
    print("Collecting all samples for batch processing...")
    all_questions = []
    all_chosen = []
    all_rejected = [] if process_rejected else None
    
    # Metadata tracking (same as original)
    flat_metadata = {
        "user_ids": [],
        "user_indices": [],
        "prompt_indices": [],
        "prompts": [],
        "chosen_responses": [],
        "rejected_responses": [] if process_rejected else None
    }
    
    for user_idx, user_data in enumerate(data["user_data"]):
        user_id = user_data["user_id"]
        questions = user_data["prompts"]
        chosen = user_data["chosen"]
        rejected = user_data["rejected"] if process_rejected else None
        
        for sample_idx, (q, c) in enumerate(zip(questions, chosen)):
            all_questions.append(q)
            all_chosen.append(c)
            if process_rejected:
                all_rejected.append(rejected[sample_idx])
            
            # Track metadata for each sample
            flat_metadata["user_ids"].append(user_id)
            flat_metadata["user_indices"].append(user_idx)
            flat_metadata["prompt_indices"].append(sample_idx)
            flat_metadata["prompts"].append(q)
            flat_metadata["chosen_responses"].append(c)
            if process_rejected:
                flat_metadata["rejected_responses"].append(rejected[sample_idx])
    
    total_samples = len(all_questions)
    print(f"Collected {total_samples} total samples for batch processing")
    
    # STEP 2: Initialize model once
    print(f"Loading scoring model: {scoring_model_name}")
    model = LLM(
        model=scoring_model_name,
        tensor_parallel_size=1,
        dtype="bfloat16" if device == "cuda" else "float32",
        gpu_memory_utilization=0.5
    )
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
    
    # STEP 3: Compute baseline ONCE for all samples
    print("Computing reference baseline for all samples...")
    base_probs_chosen, base_counts_chosen = get_log_probs(
        model, tokenizer, 
        [reference_prompt] * total_samples,
        all_questions, all_chosen, device
    )
    base_scores_chosen = torch.tensor(base_probs_chosen, device=device) / torch.tensor(base_counts_chosen, device=device)
    
    if process_rejected:
        base_probs_rejected, base_counts_rejected = get_log_probs(
            model, tokenizer,
            [reference_prompt] * total_samples, 
            all_questions, all_rejected, device
        )
        base_scores_rejected = torch.tensor(base_probs_rejected, device=device) / torch.tensor(base_counts_rejected, device=device)
    
    # STEP 4: Process each persona against ALL samples efficiently
    print(f"Processing {num_personas} personas against all {total_samples} samples...")
    chosen_scores_matrix = torch.zeros((total_samples, num_personas), device=device)
    rejected_scores_matrix = torch.zeros((total_samples, num_personas), device=device) if process_rejected else None
    
    for persona_idx, persona_prompt in enumerate(tqdm(persona_prompts, desc="Processing personas")):
        # Process chosen responses for this persona against ALL samples
        persona_probs_chosen, persona_counts_chosen = get_log_probs(
            model, tokenizer,
            [persona_prompt] * total_samples,
            all_questions, all_chosen, device
        )
        persona_scores_chosen = torch.tensor(persona_probs_chosen, device=device) / torch.tensor(persona_counts_chosen, device=device)
        
        # Compute drift scores (persona vs reference)
        chosen_scores_matrix[:, persona_idx] = persona_scores_chosen - base_scores_chosen
        
        if process_rejected:
            # Process rejected responses for this persona against ALL samples
            persona_probs_rejected, persona_counts_rejected = get_log_probs(
                model, tokenizer,
                [persona_prompt] * total_samples,
                all_questions, all_rejected, device
            )
            persona_scores_rejected = torch.tensor(persona_probs_rejected, device=device) / torch.tensor(persona_counts_rejected, device=device)
            
            # Compute drift scores for rejected
            rejected_scores_matrix[:, persona_idx] = persona_scores_rejected - base_scores_rejected
    
    # Convert to numpy (same 2D format as original)
    Y_chosen = chosen_scores_matrix.cpu().numpy()
    Y_rejected = rejected_scores_matrix.cpu().numpy() if process_rejected else None
    
    print(f"\nFinal matrix shapes:")
    print(f"  Chosen: {Y_chosen.shape}")
    if Y_rejected is not None:
        print(f"  Rejected: {Y_rejected.shape}")
    print(f"  Total samples: {total_samples}")
    print(f"  Personas (as attributes): {num_personas}")
    
    # Create metadata
    metadata = {
        "scoring_model": scoring_model_name,
        "reference_prompt": reference_prompt,
        "scoring_method": "drift_based_personas_as_attributes_efficient",
        "num_attributes": num_personas,
        "num_users": num_users,
        "num_personas": num_personas,
        "total_entries": total_samples,
        "users": data["users"],
        "persona_prompts": persona_prompts,
        "user_ids": flat_metadata["user_ids"],
        "user_indices": flat_metadata["user_indices"],
        "prompt_indices": flat_metadata["prompt_indices"],
        "prompts": flat_metadata["prompts"],
        "chosen_responses": flat_metadata["chosen_responses"],
        "rejected_responses": flat_metadata["rejected_responses"] if process_rejected else None,
        "attribute_prompts": persona_prompts,  # personas are the attributes
        "Y_chosen_shape": Y_chosen.shape,
        "Y_rejected_shape": Y_rejected.shape if process_rejected else None,
        "dtype": str(Y_chosen.dtype),
        "description": "2D matrices: Y[sample, persona] where each column is drift score for a persona vs reference. Computed with efficient batching."
    }
    
    return Y_chosen, Y_rejected, metadata


def save_flexible_reward_matrix(Y_chosen: np.ndarray, Y_rejected: Optional[np.ndarray], 
                                metadata: Dict, output_file: str):
    """Save reward matrices and metadata to compressed numpy file."""
    
    save_dict = {
        "Y": Y_chosen,  # Keep name for compatibility
        "Y_chosen": Y_chosen,
        "user_ids": np.array(metadata["user_ids"], dtype=object),
        "user_indices": np.array(metadata["user_indices"]),
        "prompt_indices": np.array(metadata["prompt_indices"]),
        "prompts": np.array(metadata["prompts"], dtype=object),
        "chosen_responses": np.array(metadata["chosen_responses"], dtype=object),
        "users": np.array(metadata["users"], dtype=object),
        "persona_prompts": np.array(metadata["persona_prompts"], dtype=object),
        "attribute_prompts": np.array(metadata["attribute_prompts"], dtype=object),
        "metadata": np.array([metadata], dtype=object)[0]
    }
    
    if Y_rejected is not None:
        save_dict["Y_rejected"] = Y_rejected
        save_dict["rejected_responses"] = np.array(metadata["rejected_responses"], dtype=object)
    
    # Save as compressed numpy archive
    np.savez_compressed(output_file, **save_dict)


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


def main():
    parser = argparse.ArgumentParser(description="Efficiently compute reward matrix from user preference data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/persona_pref",
        help="Directory containing *_train.json files"
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        default="src/core/sampled_personas.json",
        help="Path to sampled_personas.json file"
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
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name for computing log probabilities"
    )
    parser.add_argument(
        "--reference-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Reference prompt (for baseline computation)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Maximum number of users to process"
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        help="Maximum number of samples per user"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        help="Maximum number of personas to use for scoring"
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
    
    # Compute reward matrices efficiently
    Y_chosen, Y_rejected, metadata = compute_flexible_reward_matrix_efficient(
        data_dir=args.data_dir,
        personas_file=args.personas_file,
        scoring_model_name=args.scoring_model,
        reference_prompt=args.reference_prompt,
        max_users=args.max_users,
        max_samples_per_user=args.max_samples_per_user,
        max_personas=args.max_personas,
        device=args.device,
        process_rejected=not args.no_rejected
    )
    
    # Save results
    save_flexible_reward_matrix(Y_chosen, Y_rejected, metadata, args.output_file)
    
    print(f"\nEfficient reward matrix computation complete!")
    print(f"Load with: Y_chosen, Y_rejected, metadata = load_flexible_reward_matrix('{args.output_file}')")


if __name__ == "__main__":
    main()