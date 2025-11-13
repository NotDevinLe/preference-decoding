import torch
from collections import Counter

data = torch.load("eval_rewards/llama8b/original_prompts_high_entropy/train/user0.pt")

print(Counter(data['attr_scores_chosen']))