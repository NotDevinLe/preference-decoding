import json
import matplotlib.pyplot as plt
import numpy as np

# Paths to the BON JSONL files
bon_files = {
    'drift_bon': 'drift_bon.jsonl',
    'mle_bon': 'mle_bon.jsonl',
    'rm_bon_1b': 'rm_bon_1b.jsonl'
}

# Read and plot data from each file
plt.figure(figsize=(10, 6))

for method_name, file_path in bon_files.items():
    training_sizes = []
    avg_gold_selected = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            training_sizes.append(data['training_size'])
            avg_gold_selected.append(data['avg_gold_selected'])
    
    # Sort by training size for a nicer plot
    training_sizes, avg_gold_selected = zip(*sorted(zip(training_sizes, avg_gold_selected)))
    
    plt.plot(training_sizes, avg_gold_selected, marker='o', label=method_name, linewidth=2, markersize=6)

plt.xlabel('Training Size')
plt.ylabel('Average Gold Selected')
plt.title('BON Results: Training Size vs Average Gold Selected')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bon_training_size_vs_avg_gold.png')
plt.show()