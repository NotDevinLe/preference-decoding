import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_data(jsonl_file):
    """Load data from JSONL file and group by user"""
    user_data = defaultdict(list)
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                user = data['user']
                user_data[user].append({
                    'k': data['k'],
                    'acc': data['acc'],
                    'valid': data['valid'],
                    'a_rate': data['a_rate']
                })
    
    return user_data

def plot_all_users_subplots(user_data, output_file='plots/all_users_subplots.png'):
    """Create subplots for all users in a single figure"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    users = list(user_data.keys())
    n_users = len(users)
    
    # Calculate subplot layout (2 columns, enough rows)
    n_cols = 2
    n_rows = (n_users + 1) // 2  # Round up division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, user in enumerate(users):
        data = user_data[user]
        # Sort by k value
        data.sort(key=lambda x: x['k'])
        
        k_values = [d['k'] for d in data]
        accuracies = [d['acc'] for d in data]
        valid_counts = [d['valid'] for d in data]
        
        # Plot on subplot
        ax = axes_flat[i]
        ax.plot(k_values, accuracies, 'o-', linewidth=2, markersize=6, color='blue')
        
        # Add annotations for valid counts
        for k, acc, valid in zip(k_values, accuracies, valid_counts):
            ax.annotate(f'n={valid}', (k, acc), textcoords="offset points", 
                       xytext=(0,8), ha='center', fontsize=7)
        
        ax.set_xlabel('k (Number of examples)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{user}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, max(accuracies) * 1.1 if max(accuracies) > 0 else 0.1)
    
    # Hide unused subplots
    for i in range(n_users, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Accuracy vs k for All Users', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved subplots plot: {output_file}")

def plot_all_users_comparison(user_data, output_file='plots/all_users_comparison.png'):
    """Create a comparison plot showing all users on the same graph"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(user_data)))
    
    for i, (user, data) in enumerate(user_data.items()):
        # Sort by k value
        data.sort(key=lambda x: x['k'])
        
        k_values = [d['k'] for d in data]
        accuracies = [d['acc'] for d in data]
        
        plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=6, 
                label=user, color=colors[i])
    
    plt.xlabel('k (Number of examples)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k - All Users Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([2, 4, 8, 16, 25])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_file}")

def print_summary_stats(user_data):
    """Print summary statistics for each user"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for user, data in user_data.items():
        data.sort(key=lambda x: x['k'])
        
        print(f"\n{user}:")
        print("-" * 20)
        for d in data:
            print(f"  k={d['k']:2d}: acc={d['acc']:.3f}, valid={d['valid']:3d}, a_rate={d['a_rate']:.3f}")
        
        # Find best k
        best_k_data = max(data, key=lambda x: x['acc'])
        print(f"  Best k: {best_k_data['k']} (acc={best_k_data['acc']:.3f})")

if __name__ == "__main__":
    # Load data
    jsonl_file = "group1.jsonl"
    user_data = load_data(jsonl_file)
    
    print(f"Loaded data for {len(user_data)} users: {list(user_data.keys())}")
    
    # Create subplots for all users in one figure
    plot_all_users_subplots(user_data)
    
    # Create comparison plot
    plot_all_users_comparison(user_data)
    
    # Print summary statistics
    print_summary_stats(user_data)
    
    print(f"\nAll plots saved in 'plots/' directory")
