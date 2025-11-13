import torch

data1 = torch.load("eval_rewards/llama1b/persona/train/user0.pt")
data2 = torch.load("rewards/rewards_persona_testing/user0.pt")

print(data1['attr_scores_chosen'][0])
print(data2['attr_scores_chosen'][0])
print(data1 == data2)