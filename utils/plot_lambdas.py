import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('../results/reg_p.jsonl', 'r') as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Extract lambda_reg and p vectors
lambdas = []
p_list = []
for entry in entries:
    lambdas.append(entry['lambda_reg'])
    p = np.array(entry['p'])
    p_list.append(p)

# Sort by lambda
sorted_pairs = sorted(zip(lambdas, p_list))
lambdas, p_list = zip(*sorted_pairs)

# Convert to array for easier plotting
p_array = np.stack(p_list)  # shape: (num_lambdas, p_dim)

plt.figure(figsize=(12, 7))
for i in range(p_array.shape[1]):
    plt.plot(lambdas, p_array[:, i], label=f'p[{i}]')
plt.xscale('log')
plt.xlabel('lambda (log scale)')
plt.ylabel('p element value')
plt.title('Each element of p vs lambda')
plt.grid(True)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='p index')
plt.savefig('lambda_vs_p_elements.png', bbox_inches='tight')
# plt.show()
