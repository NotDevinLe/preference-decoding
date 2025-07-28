import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# File path
path = '../results/approximation_accuracy.jsonl'

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

# Group data by user and lambda
user_lambda_data = defaultdict(lambda: defaultdict(lambda: {'n': [], 'accuracy': []}))

for entry in entries:
    user = entry['user']
    lambda_val = entry['lambda']
    n = entry['n']
    accuracy = entry['accuracy']
    
    user_lambda_data[user][lambda_val]['n'].append(n)
    user_lambda_data[user][lambda_val]['accuracy'].append(accuracy)

# Sort each user-lambda combination's data by n
for user in user_lambda_data:
    for lambda_val in user_lambda_data[user]:
        sorted_pairs = sorted(zip(user_lambda_data[user][lambda_val]['n'], 
                                 user_lambda_data[user][lambda_val]['accuracy']))
        user_lambda_data[user][lambda_val]['n'], user_lambda_data[user][lambda_val]['accuracy'] = zip(*sorted_pairs)

# Define colors and line styles for visual distinction
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# Create plots for each user
for user in sorted(user_lambda_data.keys()):
    plt.figure(figsize=(12, 7))
    
    # Plot each lambda as a separate line for this user
    lambda_vals = sorted(user_lambda_data[user].keys())
    for i, lambda_val in enumerate(lambda_vals):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        
        plt.plot(user_lambda_data[user][lambda_val]['n'], 
                 user_lambda_data[user][lambda_val]['accuracy'], 
                 label=f'λ = {lambda_val}',
                 color=color,
                 linestyle=line_style,
                 marker=marker,
                 markersize=6,
                 linewidth=2)

    plt.xlabel('n (sample size)')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Sample Size for Different Lambda Values - {user}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lambda', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'../results/approximation_accuracy_by_lambda_{user}.png', bbox_inches='tight', dpi=300)
    plt.show()

# Optional: Create a combined plot showing all users
if len(user_lambda_data) > 1:
    plt.figure(figsize=(15, 10))
    
    plot_idx = 0
    for user in sorted(user_lambda_data.keys()):
        lambda_vals = sorted(user_lambda_data[user].keys())
        for lambda_val in lambda_vals:
            color = colors[plot_idx % len(colors)]
            line_style = line_styles[plot_idx % len(line_styles)]
            marker = markers[plot_idx % len(markers)]
            
            plt.plot(user_lambda_data[user][lambda_val]['n'], 
                     user_lambda_data[user][lambda_val]['accuracy'], 
                     label=f'{user}, λ = {lambda_val}',
                     color=color,
                     linestyle=line_style,
                     marker=marker,
                     markersize=5,
                     linewidth=1.5,
                     alpha=0.8)
            plot_idx += 1

    plt.xlabel('n (sample size)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample Size - All Users and Lambda Values')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='User, Lambda', fontsize=8)
    plt.tight_layout()
    plt.savefig('../results/approximation_accuracy_all_users.png', bbox_inches='tight', dpi=300)
    plt.show()

# Print summary statistics
print("Summary Statistics:")
print("=" * 50)
for user in sorted(user_lambda_data.keys()):
    print(f"\n{user}:")
    for lambda_val in sorted(user_lambda_data[user].keys()):
        accuracies = user_lambda_data[user][lambda_val]['accuracy']
        print(f"  λ = {lambda_val}: avg accuracy = {np.mean(accuracies):.3f}, "
              f"max accuracy = {np.max(accuracies):.3f}")