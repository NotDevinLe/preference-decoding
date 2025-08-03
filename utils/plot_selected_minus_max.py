import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# File paths (corrected)
files = [
    '/gscratch/ark/devinl6/preference/preference-decoding/results/rm_bon_by_n.jsonl',
    '/gscratch/ark/devinl6/preference/preference-decoding/results/approx_bon_by_n.jsonl',
]

# Labels for each file
labels = [
    'RM BON',
    'Approx BON'
]

# Colors and styles for each line
colors = ['blue', 'red', 'green']
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

# Load and process data for each file
file_data = {}

for i, (file_path, label) in enumerate(zip(files, labels)):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    print(f"Loading data from: {file_path}")
    
    # Load data
    entries = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {file_path}: {e}")
                    print(f"Line content: {line.strip()}")
    
    print(f"Loaded {len(entries)} entries from {label}")
    
    if len(entries) == 0:
        print(f"No valid entries found in {file_path}")
        continue
    
    # Print first few entries to debug
    print(f"First few entries from {label}:")
    for j, entry in enumerate(entries[:3]):
        print(f"  Entry {j+1}: {entry}")
    
    # Group by k (not n) and calculate average selected_minus_max
    k_data = defaultdict(list)
    for entry in entries:
        # Handle both 'k' and 'n' keys
        k = entry.get('k', entry.get('n'))
        if k is None:
            print(f"Warning: No 'k' or 'n' key found in entry: {entry}")
            continue
            
        # Handle both 'selected_minus_max' and 'avg_selected_minus_max'
        selected_minus_max = entry.get('selected_minus_max', entry.get('avg_selected_minus_max'))
        if selected_minus_max is None:
            print(f"Warning: No 'selected_minus_max' or 'avg_selected_minus_max' key found in entry: {entry}")
            continue
            
        k_data[k].append(selected_minus_max)
    
    # Calculate average for each k
    k_values = []
    avg_selected_minus_max = []
    for k in sorted(k_data.keys()):
        k_values.append(k)
        avg_selected_minus_max.append(sum(k_data[k]) / len(k_data[k]))
    
    print(f"Processed k values: {k_values}")
    print(f"Average selected_minus_max values: {avg_selected_minus_max}")
    
    file_data[label] = {
        'k': k_values,
        'selected_minus_max': avg_selected_minus_max,
        'color': colors[i],
        'line_style': line_styles[i],
        'marker': markers[i]
    }

# Check if we have any data to plot
if not file_data:
    print("No data found to plot!")
    exit(1)

# Create plot
plt.figure(figsize=(12, 8))

for label, data in file_data.items():
    if len(data['k']) == 0:
        print(f"No data points for {label}")
        continue
        
    plt.plot(data['k'], 
             data['selected_minus_max'], 
             label=label,
             color=data['color'],
             linestyle=data['line_style'],
             marker=data['marker'],
             markersize=8,
             linewidth=2,
             markerfacecolor='white',
             markeredgewidth=2)

# Add horizontal line at y=2.8970 (Random baseline)

plt.xlabel('k (Best-of-k)', fontsize=12)
plt.ylabel('Selected Minus Max (Gap from Optimal)', fontsize=12)
plt.title('Best-of-N Performance: Gap from Optimal vs k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Set reasonable axis limits
if file_data:
    all_k_values = []
    all_smm_values = []
    for data in file_data.values():
        all_k_values.extend(data['k'])
        all_smm_values.extend(data['selected_minus_max'])
    
    if all_k_values and all_smm_values:
        plt.xlim(min(all_k_values) - 1, max(all_k_values) + 1)
        plt.ylim(min(all_smm_values) - 0.5, max(all_smm_values) + 0.5)

plt.tight_layout()

# Save with absolute path
output_path = '/gscratch/ark/devinl6/preference/preference-decoding/results/selected_minus_max_comparison_by_k.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Plot saved to: {output_path}")

plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 60)
for label, data in file_data.items():
    print(f"\n{label}:")
    if len(data['k']) == 0:
        print("  No data available")
        continue
        
    for k, smm in zip(data['k'], data['selected_minus_max']):
        improvement = 2.8970 - smm  # How much better than random
        print(f"  k={k:2d}: selected_minus_max = {smm:6.4f} (improvement over random: {improvement:+6.4f})")
    
    # Find best k for this method
    if data['selected_minus_max']:
        best_idx = data['selected_minus_max'].index(min(data['selected_minus_max']))
        best_k = data['k'][best_idx]
        best_smm = data['selected_minus_max'][best_idx]
        print(f"  Best k: {best_k} (selected_minus_max = {best_smm:.4f})")

# Compare methods if we have both
if len(file_data) >= 2:
    print(f"\n{'Method Comparison:'}")
    print("=" * 60)
    methods = list(file_data.keys())
    method1, method2 = methods[0], methods[1]
    
    # Find common k values
    common_k = set(file_data[method1]['k']) & set(file_data[method2]['k'])
    
    if common_k:
        print(f"Comparing {method1} vs {method2} at common k values:")
        for k in sorted(common_k):
            idx1 = file_data[method1]['k'].index(k)
            idx2 = file_data[method2]['k'].index(k)
            smm1 = file_data[method1]['selected_minus_max'][idx1]
            smm2 = file_data[method2]['selected_minus_max'][idx2]
            diff = smm1 - smm2
            winner = method1 if smm1 < smm2 else method2  # Lower is better
            print(f"  k={k:2d}: {method1} = {smm1:.4f}, {method2} = {smm2:.4f}, diff = {diff:+.4f} (winner: {winner})")