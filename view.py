import torch

data = torch.load("rewards_low_temp_original/reward_matrices_9.pt")
print((data['attr_counts_chosen'] == 0).sum())