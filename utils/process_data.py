import torch
import numpy as np


def l1_solve(d_mean, l1_lambda, std=None):
    d = np.asarray(d_mean, dtype=float)
    z = np.sign(d) * np.maximum(np.abs(d) - l1_lambda, 0.0)
    norm = np.linalg.norm(z, ord=2)
    if norm == 0.0:
        return np.zeros_like(d)
    if std is None:
        return z / norm
    else:
        return z / (norm * std)

def approximate(X: torch.Tensor, l1_lambda: float = 0.01) -> np.ndarray:
    col_std = X.std(dim=0).clamp_min(1e-8)
    d = (X / col_std).mean(dim=0).detach().cpu().numpy()
    
    p = l1_solve(d, l1_lambda, std=col_std.detach().cpu().numpy())
    
    return p

selected_attributes = {
    'k2': list(set([99, 197])),
    'k5': list(set([189, 127, 38, 104, 62])),
    'k10': list(set([126, 169, 43, 199, 21, 147, 168, 45, 72, 198])),
    'k20': list(set([191, 74, 42, 2, 29, 114, 90, 89, 153, 187, 108, 75, 152, 199, 25, 59, 110, 104, 136, 168])),
    'k30': list(set([169, 44, 60, 191, 1, 101, 52, 152, 187, 147, 198, 95, 126, 89, 13, 199, 168, 83, 151, 61, 153, 136, 59, 16, 139, 190, 30, 104, 175, 78]))
}

for k in selected_attributes.keys():
    total_correct = 0
    total_samples = 0
    
    for user_id in range(120, 150):
        rewards = torch.load(f'rewards_persona_testing_small/user{user_id}.pt')
        chosen = rewards['attr_scores_chosen'] / rewards['attr_counts_chosen'].clamp(min=1e-9) - rewards['base_scores_chosen'].unsqueeze(1) / rewards['base_counts_chosen'].unsqueeze(1).clamp(min=1e-9)
        rejected = rewards['attr_scores_rejected'] / rewards['attr_counts_rejected'].clamp(min=1e-9) - rewards['base_scores_rejected'].unsqueeze(1) / rewards['base_counts_rejected'].unsqueeze(1).clamp(min=1e-9)
        
        # Select only the attributes for this k value
        chosen = chosen[:, selected_attributes[k]]
        rejected = rejected[:, selected_attributes[k]]
        
        # Compute preference differences
        X = chosen - rejected
        
        # Fit the model to get weights
        p = approximate(X, l1_lambda=0.01)
        
        # Apply the learned weights to get final scores
        chosen_scores = (chosen * p).sum(dim=1)
        rejected_scores = (rejected * p).sum(dim=1)
        
        # Count correct predictions (chosen should be higher than rejected)
        correct = (chosen_scores > rejected_scores).sum().item()
        total_correct += correct
        total_samples += len(chosen_scores)

    print(f"k={k}: Total correct: {total_correct}")
    print(f"k={k}: Total samples: {total_samples}")
    print(f"k={k}: Accuracy: {total_correct / total_samples if total_samples > 0 else 0:.4f}")
    print("-" * 50)