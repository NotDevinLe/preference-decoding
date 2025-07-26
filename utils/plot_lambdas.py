import json
import numpy as np
import matplotlib.pyplot as plt
import os

# File path
path = '../results/l1_reg/user2_l1_reg.jsonl'

# Load results
entries = []
if os.path.exists(path):
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
else:
    print(f"File not found: {path}")
    exit(1)

# Extract log(lambda) and p vectors
log_lambdas = []
p_list = []
for entry in entries:
    log_lambdas.append(np.log(entry['lambda']))
    p = np.array(entry['p'])
    p_list.append(p)

# Sort by log(lambda)
sorted_pairs = sorted(zip(log_lambdas, p_list))
log_lambdas, p_list = zip(*sorted_pairs)

# Convert to array for easier plotting
p_array = np.stack(p_list)  # shape: (num_lambdas, p_dim)

plt.figure(figsize=(12, 7))
for i in range(p_array.shape[1]):
    plt.plot(log_lambdas, p_array[:, i], label=f'p[{i}]')
plt.xlabel('log(lambda)')
plt.ylabel('p element value')
plt.title('Each element of p vs log(lambda)')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='p index', fontsize=9)
plt.savefig('../results/l1_reg/user2_l1_reg_lambda_vs_p_elements.png', bbox_inches='tight')
plt.show()
