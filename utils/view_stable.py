import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Load non-stable results
with open('../results/user_p.jsonl', 'r') as f:
    non_stable_data = [json.loads(line) for line in f if line.strip()]

# Load stable results
with open('../results/user_p_stable.jsonl', 'r') as f:
    stable_data = [json.loads(line) for line in f if line.strip()]

# Organize data by user: n -> mean(accuracy)
def get_user_n_to_mean(data):
    user_n_to_mean = defaultdict(dict)
    for entry in data:
        user = entry['user']
        n = entry['n']
        p = entry['p']
        mean_acc = np.mean(p)
        user_n_to_mean[user][n] = mean_acc
    return user_n_to_mean

non_stable_means = get_user_n_to_mean(non_stable_data)
stable_means = get_user_n_to_mean(stable_data)

users = sorted(set(list(non_stable_means.keys()) + list(stable_means.keys())))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, user in enumerate(users):
    ax = axes[idx]
    # Get sorted n values for each version
    ns_n = sorted(non_stable_means[user].keys())
    st_n = sorted(stable_means[user].keys())
    # X: n, Y: mean accuracy
    ns_x = ns_n
    ns_y = [non_stable_means[user][n] for n in ns_n]
    st_x = st_n
    st_y = [stable_means[user][n] for n in st_n]
    ax.plot(ns_x, ns_y, 'o--', label='Non-stable')
    ax.plot(st_x, st_y, 'o-', label='Stable')
    ax.set_title(f'User: {user}')
    ax.set_xlabel('n')
    ax.set_ylabel('Mean Accuracy')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('user_comparison.png')
# plt.show()
