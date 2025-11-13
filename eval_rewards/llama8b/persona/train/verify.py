import torch

data = torch.load("eval_rewards/llama8b/persona/train/user0.pt")

print(data['attr_scores_chosen'])