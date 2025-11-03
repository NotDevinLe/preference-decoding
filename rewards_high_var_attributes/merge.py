import torch
import glob

# Find all user*.pt files in current directory
user_files = sorted(glob.glob("user*.pt"))

print(f"Found {len(user_files)} user files to merge")

# Load and concatenate all user reward tensors
all_rewards = []
for user_file in user_files:
    print(f"Loading {user_file}...")
    user_rewards = torch.load(user_file)

    chosen_rewards = user_rewards['attr_scores_chosen'] / user_rewards['attr_counts_chosen'].clamp(min=1e-9) - user_rewards['base_scores_chosen'].unsqueeze(1) / user_rewards['base_counts_chosen'].unsqueeze(1).clamp(min=1e-9)
    rejected_rewards = user_rewards['attr_scores_rejected'] / user_rewards['attr_counts_rejected'].clamp(min=1e-9) - user_rewards['base_scores_rejected'].unsqueeze(1) / user_rewards['base_counts_rejected'].unsqueeze(1).clamp(min=1e-9)
    all_rewards.append(chosen_rewards - rejected_rewards)

# Concatenate all rewards
merged_rewards = torch.cat(all_rewards, dim=0)

print(f"Merged rewards shape: {merged_rewards.shape}")
print(f"Saving to ../data/rewards.pt")

# Save the merged rewards
torch.save(merged_rewards, "rewards.pt")

print(f"Successfully merged {len(user_files)} user reward files")