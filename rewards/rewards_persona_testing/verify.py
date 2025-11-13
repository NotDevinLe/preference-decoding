import torch

data = torch.load("rewards/rewards_persona_testing/user0.pt")
print(data['attr_scores_chosen'][:10])