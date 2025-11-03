import torch

total = torch.load('rewards/reward_matrices_0.pt')

for i in range(1, 10):
    total = torch.cat([total, torch.load(f'rewards/reward_matrices_{i}.pt')], dim=0)

print(total.shape)
torch.save(total, 'data/toy_rewards.pt')