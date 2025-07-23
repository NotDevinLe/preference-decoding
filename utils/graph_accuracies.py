import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load drift approximation data
drift_df = pd.read_json("../results/approximation_results.jsonl", lines=True)

# Exclude user 4 from drift approximation data
drift_df = drift_df[drift_df["user"] != "user4"]

# Load reward model data
reward_df = pd.read_json("../results/reward_results.jsonl", lines=True)

# Group by 'n' and compute mean accuracy for drift approximation
# (mean across users for each sample size)
drift_grouped = drift_df.groupby("n")["acc"].mean().reset_index()
drift_grouped = drift_grouped.sort_values("n")

# Group by 'n' and compute mean accuracy for reward model
reward_grouped = reward_df.groupby("n")["acc"].mean().reset_index()
reward_grouped = reward_grouped.sort_values("n")

# Create the plot
plt.figure(figsize=(10, 6))

# Drift Approximation Plot (mean across users)
plt.plot(drift_grouped["n"], drift_grouped["acc"], marker='o', label="Drift Approximation (mean, excl. user4)", linewidth=2, markersize=6)

# Reward Model Plot (mean across users)
plt.plot(reward_grouped["n"], reward_grouped["acc"], marker='s', linestyle='--', 
         label="Reward Model (mean)", linewidth=2, markersize=6, color='red')

# Labels and title
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("Average Accuracy", fontsize=12)
plt.title("Average Accuracy vs Sample Size: Drift Approximation (excl. user4) vs Reward Model", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

# Set axis limits for better visualization
plt.xlim(0, max(max(drift_grouped["n"]), max(reward_grouped["n"])) + 10)

# Add some styling
plt.tight_layout()

# Save the plot
plt.savefig("accuracy_comparison_mean_excl_user4.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Drift Approximation Summary (mean across users, excl. user4):")
print(drift_grouped)

print("\nReward Model Summary (mean across users):")
print(reward_grouped)

