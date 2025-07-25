import matplotlib.pyplot as plt
import json
import argparse
import os

parser = argparse.ArgumentParser()
# No user argument needed, always user1 for this plot
args = parser.parse_args()
user = 'user1'

# 1. Persona approx
persona_path = '../results/approx_bon_persona_by_n.jsonl'
persona_data = []
if os.path.exists(persona_path):
    with open(persona_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('user') == user:
                persona_data.append(entry)
else:
    print(f"File not found: {persona_path}")

# 2. Normal approx
normal_path = '../results/approx_bon_by_n.jsonl'
normal_data = []
if os.path.exists(normal_path):
    with open(normal_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('user') == user:
                normal_data.append(entry)
else:
    print(f"File not found: {normal_path}")

# 3. Reward model
reward_path = '../results/rm_bon_by_n.jsonl'
reward_data = []
if os.path.exists(reward_path):
    with open(reward_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('user') == user:
                reward_data.append(entry)
else:
    print(f"File not found: {reward_path}")

# 4. Random selection
random_path = '../results/approx_bon_by_n_random.jsonl'
random_data = []
if os.path.exists(random_path):
    with open(random_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('user') == user:
                random_data.append(entry)
else:
    print(f"File not found: {random_path}")

# Sort by k for plotting
persona_data = sorted(persona_data, key=lambda x: x['k'])
normal_data = sorted(normal_data, key=lambda x: x['k'])
reward_data = sorted(reward_data, key=lambda x: x['k'])
random_data = sorted(random_data, key=lambda x: x['k'])

plt.figure(figsize=(8, 5))
if persona_data:
    plt.plot([row['k'] for row in persona_data], [row['avg_selected_minus_max'] for row in persona_data], marker='o', label='persona approx')
if normal_data:
    plt.plot([row['k'] for row in normal_data], [row['avg_selected_minus_max'] for row in normal_data], marker='s', label='normal approx')
if reward_data:
    plt.plot([row['k'] for row in reward_data], [row['avg_selected_minus_max'] for row in reward_data], marker='^', label='reward model')
if random_data:
    plt.plot([row['k'] for row in random_data], [row['avg_selected_minus_max'] for row in random_data], marker='d', label='random')

plt.xlabel('k')
plt.ylabel('avg_selected_minus_max')
plt.title('Avg Selected Minus Max vs k (user1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bon_selected_minus_max_vs_k_user1.png', dpi=200)
plt.show()
