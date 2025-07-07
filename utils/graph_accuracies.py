import pandas as pd
import matplotlib.pyplot as plt

# Load JSONL data
df = pd.read_json("../approximation_results.jsonl", lines=True)

# Group by 'n' and compute mean accuracy
grouped = df.groupby("n")["acc"].mean().reset_index()

# Scale n to 0.8n
grouped = grouped.sort_values("n")

# Drift Approximation Plot
plt.plot(grouped["n"], grouped["acc"], marker='o', label="Drift Approximation")

# Reward Model data (scaled n, manually specified)
reward_n = [128, 64, 32, 16]
reward_acc = [0.86, 0.74, 0.68, 0.74]
plt.plot(reward_n, reward_acc, marker='s', linestyle='--', label="Reward Model")

# Labels and title
plt.xlabel("Sample Size")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy vs Sample Size")
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig("accuracy_vs_n.png")

