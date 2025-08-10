import json
import matplotlib.pyplot as plt
import numpy as np

# Paths to the BON by n JSONL files
bon_files = {
    'rm_bon_by_n_1b': 'rm_bon_by_n_1b.jsonl',
    'approx_bon_by_n': 'drift_bon_by_n.jsonl',
    'mle_bon_by_n': 'mle_bon_by_n.jsonl'
}

# Read and plot data from each file
plt.figure(figsize=(10, 6))

for method_name, file_path in bon_files.items():
    k_values = []
    avg_selected_gold = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            k_values.append(data['k'])
            avg_selected_gold.append(data['avg_selected_gold'])
    
    # Sort by k for a nicer plot
    k_values, avg_selected_gold = zip(*sorted(zip(k_values, avg_selected_gold)))
    
    plt.plot(k_values, avg_selected_gold, marker='o', label=method_name, linewidth=2, markersize=6)

plt.xlabel('k (Number of Outputs)')
plt.ylabel('Average Selected Gold')
plt.title('BON Results: k vs Average Selected Gold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bon_k_vs_avg_selected_gold.png')
plt.show()
