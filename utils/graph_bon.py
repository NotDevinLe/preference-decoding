import pandas as pd
import matplotlib.pyplot as plt

# Load drift approximation BON data
drift_df = pd.read_json("../results/approx_bon.jsonl", lines=True)

# Load reward model BON data
reward_df = pd.read_json("../results/rm_bon.jsonl", lines=True)

# Group by 'n' and compute mean selected_minus_max for drift approximation
drift_grouped = drift_df.groupby("n")["selected_minus_max"].mean().reset_index()
drift_grouped = drift_grouped.sort_values("n")

# Group by 'n' and compute mean selected_minus_max for reward model
reward_grouped = reward_df.groupby("n")["selected_minus_max"].mean().reset_index()
reward_grouped = reward_grouped.sort_values("n")

# Create the plot
plt.figure(figsize=(10, 6))

# Drift Approximation Plot
plt.plot(drift_grouped["n"], drift_grouped["selected_minus_max"], marker='o', label="Drift Approximation (Selected - Max)", linewidth=2, markersize=6)

# Reward Model Plot
plt.plot(reward_grouped["n"], reward_grouped["selected_minus_max"], marker='s', linestyle='--', label="Reward Model (Selected - Max)", linewidth=2, markersize=6)

# Labels and title
plt.xlabel("Sample Size", fontsize=12)
plt.ylabel("Selected - Max Gold Reward (lower is better)", fontsize=12)
plt.title("Selected vs Max Gold Reward: Drift Approximation vs Reward Model (Best-of-N)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Set axis limits for better visualization
plt.xlim(0, max(max(drift_grouped["n"]), max(reward_grouped["n"])) + 10)
# Set y-axis limits based on data range
all_diffs = list(drift_grouped["selected_minus_max"]) + list(reward_grouped["selected_minus_max"])
y_min, y_max = min(all_diffs), max(all_diffs)
y_range = y_max - y_min
plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

# Add some styling
plt.tight_layout()

# Save the plot
plt.savefig("bon_selected_minus_max_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Drift Approximation BON (Selected - Max) Summary:")
print(drift_grouped)
print("\nReward Model BON (Selected - Max) Summary:")
print(reward_grouped)
