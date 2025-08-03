import json
import matplotlib.pyplot as plt
import numpy as np

# Files to plot
files = [
    '../results/approx_bon_by_n.jsonl',
    '../results/rm_bon_by_n.jsonl'
]

labels = [
    'Approx BON',
    'RM BON'
]

# Colors and styles for each line
colors = ['blue', 'red']
line_styles = ['-', '--']
markers = ['o', 's']

plt.figure(figsize=(10, 6))

# Load and process data for each file
for i, (file_path, label) in enumerate(zip(files, labels)):
    n_values = []
    avg_selected_gold_values = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            n_values.append(data['k'])  # Using 'k' as n (number of outputs judged)
            avg_selected_gold_values.append(data['avg_selected_gold'])
    
    # Sort by n values
    sorted_data = sorted(zip(n_values, avg_selected_gold_values))
    n_sorted, gold_sorted = zip(*sorted_data)
    
    plt.plot(n_sorted, gold_sorted, 
             color=colors[i], 
             linestyle=line_styles[i], 
             marker=markers[i], 
             label=label,
             linewidth=2,
             markersize=6)

plt.xlabel('n (number of outputs judged)', fontsize=12)
plt.ylabel('Average Selected Gold Score', fontsize=12)
plt.title('Average Selected Gold Score vs Number of Outputs Judged', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('../results/avg_selected_gold_vs_n.png', bbox_inches='tight', dpi=300)
plt.show() 