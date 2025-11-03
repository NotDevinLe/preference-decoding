import json
import torch

parts = []

for i in range(120, 150):
    t = torch.load(f"rewards_persona_testing_small/user{i}.pt", map_location="cpu")
    chosen_score = t['attr_scores_chosen'] / t['attr_counts_chosen'].clamp(min=1e-9) - t['base_scores_chosen'].unsqueeze(1) / t['base_counts_chosen'].unsqueeze(1).clamp(min=1e-9)
    rejected_score = t['attr_scores_rejected'] / t['attr_counts_rejected'].clamp(min=1e-9) - t['base_scores_rejected'].unsqueeze(1) / t['base_counts_rejected'].unsqueeze(1).clamp(min=1e-9)

    parts.append(chosen_score - rejected_score)

X = torch.cat(parts, dim=0)

print(X.shape)
torch.save(X, "rewards_persona_testing_validation.pt")