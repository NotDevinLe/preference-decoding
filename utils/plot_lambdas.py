import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# File path
path = '../results/preference/user1_val_results.json'

# Load results
if os.path.exists(path):
    with open(path, 'r') as f:
        entries = json.load(f)
else:
    print(f"File not found: {path}")
    exit(1)

# Group data by lambda
lambda_data = defaultdict(lambda: {'n': [], 'accuracy': []})

for entry in entries:
    lambda_val = entry['lambda']
    n = entry['sample_size']  # Changed from 'n' to 'sample_size'
    accuracy = entry['accuracy']
    
    lambda_data[lambda_val]['n'].append(n)
    lambda_data[lambda_val]['accuracy'].append(accuracy)

# Sort each lambda's data by n
for lambda_val in lambda_data:
    sorted_pairs = sorted(zip(lambda_data[lambda_val]['n'], lambda_data[lambda_val]['accuracy']))
    lambda_data[lambda_val]['n'], lambda_data[lambda_val]['accuracy'] = zip(*sorted_pairs)

# Create plot
plt.figure(figsize=(12, 7))

# Define colors and line styles for visual distinction
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# Plot each lambda as a separate line
for i, lambda_val in enumerate(sorted(lambda_data.keys())):
    color = colors[i % len(colors)]
    line_style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    
    plt.plot(lambda_data[lambda_val]['n'], 
             lambda_data[lambda_val]['accuracy'], 
             label=f'λ = {lambda_val}',
             color=color,
             linestyle=line_style,
             marker=marker,
             markersize=6,
             linewidth=2)

plt.xlabel('n (sample size)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Sample Size for Different Lambda Values')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lambda', fontsize=9)
plt.tight_layout()
plt.savefig('../results/approximation_accuracy_by_lambda.png', bbox_inches='tight', dpi=300)
plt.show()