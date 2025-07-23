import json
import matplotlib.pyplot as plt

# Load results
with open('../results/approximation_results_stable.jsonl', 'r') as f:
    stable_results = [json.loads(line) for line in f if line.strip()]
with open('../results/approximation_results.jsonl', 'r') as f:
    nonstable_results = [json.loads(line) for line in f if line.strip()]

# Filter for user1
stable_user1 = [r for r in stable_results if r['user'] == 'user1']
nonstable_user1 = [r for r in nonstable_results if r['user'] == 'user1']

# Get n and acc
stable_n = [r['n'] for r in stable_user1]
stable_acc = [r['acc'] for r in stable_user1]

nonstable_n = [r['n'] for r in nonstable_user1]
nonstable_acc = [r['acc'] for r in nonstable_user1]

# Sort by n
stable_sorted = sorted(zip(stable_n, stable_acc))
nonstable_sorted = sorted(zip(nonstable_n, nonstable_acc))

stable_n, stable_acc = zip(*stable_sorted)
nonstable_n, nonstable_acc = zip(*nonstable_sorted)

plt.figure(figsize=(8,6))
plt.plot(stable_n, stable_acc, 'o-', label='Stable')
plt.plot(nonstable_n, nonstable_acc, 'o-', label='Non-stable')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.title('User 1: Stable vs Non-stable Accuracy by n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('stable_vs_nonstable_user1.png')
# plt.show()
