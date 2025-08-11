import json
import matplotlib.pyplot as plt
import numpy as np

# Path to the JSONL file
jsonl_path = '/gscratch/ark/devinl6/preference/preference-decoding/results/mle/user1_lambda.jsonl'

# Read the data
lambdas = []
p_vectors = []

with open(jsonl_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        lambdas.append(data['l1_lambda'])
        p_vectors.append(data['p_vector'])

# Sort by lambda for a nicer plot
lambdas, p_vectors = zip(*sorted(zip(lambdas, p_vectors)))

# Convert to numpy arrays for easier manipulation
lambdas = np.array(lambdas)
p_vectors = np.array(p_vectors)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each p_vector element as a separate line
for i in range(p_vectors.shape[1]):
    plt.plot(lambdas, p_vectors[:, i], marker='o', label=f'p[{i}]')

plt.xlabel('L1 Lambda')
plt.ylabel('P Vector Elements')
plt.title('P Vector Elements vs L1 Lambda')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('/gscratch/ark/devinl6/preference/preference-decoding/results/mle/lambda_vs_p_elements.png', bbox_inches='tight')
plt.show()
