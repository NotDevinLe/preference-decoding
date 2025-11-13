import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

parts = []

for i in range(10):
    t = torch.load(f"eval_rewards/llama8b/original_prompts_high_entropy/train/user{i}.pt", map_location="cpu")
    chosen_score = t['attr_scores_chosen'] / t['attr_counts_chosen'].clamp(min=1e-9) - t['base_scores_chosen'].unsqueeze(1) / t['base_counts_chosen'].unsqueeze(1).clamp(min=1e-9)
    rejected_score = t['attr_scores_rejected'] / t['attr_counts_rejected'].clamp(min=1e-9) - t['base_scores_rejected'].unsqueeze(1) / t['base_counts_rejected'].unsqueeze(1).clamp(min=1e-9)

    parts.append(chosen_score)

X = torch.cat(parts, dim=0)
mask = ~torch.isnan(X).any(dim=1)
X = X[mask]

X_centered = X - X.mean(dim=0, keepdim=True)

cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

std = cov.diag().sqrt().unsqueeze(0)
corr = cov / (std.T @ std + 1e-9)

print(corr)

# Convert to numpy for plotting
corr_np = corr.numpy()

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Heatmap of correlation matrix
sns.heatmap(corr_np, 
            annot=False, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            cbar_kws={'shrink': 0.8},
            ax=axes[0])
axes[0].set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Attributes', fontsize=12)
axes[0].set_ylabel('Attributes', fontsize=12)

# Plot 2: Distribution of correlation values
# Get upper triangular correlations (excluding diagonal)
upper_tri = np.triu(corr_np, k=1)
correlations = upper_tri[upper_tri != 0]

axes[1].hist(correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1].axvline(correlations.mean(), color='red', linestyle='--', 
                label=f'Mean: {correlations.mean():.3f}')
axes[1].axvline(np.median(correlations), color='green', linestyle='--', 
                label=f'Median: {np.median(correlations):.3f}')
axes[1].set_title('Distribution of Correlation Values', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Add statistics text
stats_text = f"""Statistics:
Total pairs: {len(correlations)}
Mean correlation: {correlations.mean():.4f}
Std correlation: {correlations.std():.4f}
Min correlation: {correlations.min():.4f}
Max correlation: {correlations.max():.4f}
Strong correlations (|r| > 0.5): {np.sum(np.abs(correlations) > 0.5)}
"""

axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print(f"Correlation analysis saved as 'correlation_analysis.png'")

# Show summary statistics
print(f"\nCorrelation Matrix Statistics:")
print(f"Shape: {corr_np.shape}")
print(f"Mean correlation: {correlations.mean():.4f}")
print(f"Std correlation: {correlations.std():.4f}")
print(f"Min correlation: {correlations.min():.4f}")
print(f"Max correlation: {correlations.max():.4f}")
print(f"Number of strong correlations (|r| > 0.5): {np.sum(np.abs(correlations) > 0.5)}")

plt.close()  # Close the figure to free memory