import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Extract expectation matrix and chosen rewards from old checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to old checkpoint file containing both expectation matrix and chosen rewards")
    parser.add_argument("--extract_both", action="store_true", help="Extract both expectation matrix and chosen rewards")
    args = parser.parse_args()
    
    # Load the old checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    
    # Check what's in the checkpoint
    print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
    
    # Extract base info from filename
    dir_name = os.path.dirname(args.checkpoint_path)
    base_name = os.path.basename(args.checkpoint_path)
    
    # Extract user, n, and size from filename (e.g., user1_expectation_n16_size200.pt)
    parts = base_name.split('_')
    user = parts[0]
    # Find n value
    n_value = None
    for part in parts:
        if part.startswith('n'):
            n_value = part[1:]  # Remove 'n' prefix
            break
    # Get size
    size = parts[-1].replace('.pt', '')  # Get the size part
    
    # Extract expectation matrix (default behavior)
    if 'expectation_matrix' in checkpoint:
        expectation_matrix = checkpoint['expectation_matrix']
        print(f"Found expectation_matrix with shape: {expectation_matrix.shape}")
        
        # Create output path for expectation matrix
        expectation_path = os.path.join(dir_name, f"{user}_expectation_n{n_value}_{size}.pt")
        
        # Save expectation matrix in new format
        print(f"Saving expectation matrix to: {expectation_path}")
        torch.save({
            'expectation_matrix': expectation_matrix,
            'num_expectation_samples': expectation_matrix.shape[1] if len(expectation_matrix.shape) > 1 else None,
            'num_attributes': expectation_matrix.shape[2] if len(expectation_matrix.shape) > 2 else None,
            'num_prompts': expectation_matrix.shape[0]
        }, expectation_path)
        print(f"Expectation matrix saved!")
    else:
        print("Error: No 'expectation_matrix' found in checkpoint!")
    
    # Extract chosen rewards only if explicitly requested
    if args.extract_both and 'chosen_rewards' in checkpoint:
        chosen_rewards = checkpoint['chosen_rewards']
        print(f"Found chosen_rewards with shape: {chosen_rewards.shape}")
        
        # Create output path for chosen rewards
        rewards_path = os.path.join(dir_name, f"{user}_chosen_rewards_{size}.pt")
        
        # Save just the chosen rewards
        print(f"Saving chosen rewards to: {rewards_path}")
        torch.save({
            'chosen_rewards': chosen_rewards,
            'num_data_points': chosen_rewards.shape[0],
            'num_attributes': chosen_rewards.shape[1]
        }, rewards_path)
        
        print(f"Chosen rewards saved!")
    else:
        print("Warning: No 'chosen_rewards' found in checkpoint!")
        print("This checkpoint may have been created with the new architecture that doesn't include chosen rewards.")
    
    print("\nExtraction complete!")
    if 'expectation_matrix' in checkpoint:
        print(f"Expectation matrix shape: {checkpoint['expectation_matrix'].shape}")
    if args.extract_both and 'chosen_rewards' in checkpoint:
        print(f"Chosen rewards shape: {checkpoint['chosen_rewards'].shape}")

if __name__ == "__main__":
    main()