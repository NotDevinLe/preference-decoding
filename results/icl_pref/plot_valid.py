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

def plot_valid_vs_k_all_users(user_data, output_file='plots/valid_vs_k_all_users.png'):
    """Create a plot showing valid samples vs k for all users"""
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(user_data)))
    
    for i, (user, data) in enumerate(user_data.items()):
        # Sort by k value
        data.sort(key=lambda x: x['k'])
        
        k_values = [d['k'] for d in data]
        valid_counts = [d['valid'] for d in data]
        
        plt.plot(k_values, valid_counts, 'o-', linewidth=2, markersize=6, 
                label=user, color=colors[i])
    
    plt.xlabel('k (Number of examples)')
    plt.ylabel('Number of Valid Samples')
    plt.title('Number of Valid Samples vs k - All Users')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([2, 4, 8, 16, 25])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved valid vs k plot: {output_file}")

def plot_valid_vs_k_subplots(user_data, output_file='plots/valid_vs_k_subplots.png'):
    """Create subplots for all users showing valid samples vs k"""
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
        valid_counts = [d['valid'] for d in data]
        
        # Plot on subplot
        ax = axes_flat[i]
        ax.plot(k_values, valid_counts, 'o-', linewidth=2, markersize=6, color='green')
        
        # Add annotations for valid counts
        for k, valid in zip(k_values, valid_counts):
            ax.annotate(f'{valid}', (k, valid), textcoords="offset points", 
                       xytext=(0,8), ha='center', fontsize=7)
        
        ax.set_xlabel('k (Number of examples)')
        ax.set_ylabel('Valid Samples')
        ax.set_title(f'{user}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, max(valid_counts) * 1.1 if max(valid_counts) > 0 else 10)
    
    # Hide unused subplots
    for i in range(n_users, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Number of Valid Samples vs k for All Users', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved valid vs k subplots: {output_file}")

def plot_individual_valid_plots(user_data, output_dir='plots'):
    """Create individual plots for each user showing valid samples vs k"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for user, data in user_data.items():
        # Sort by k value
        data.sort(key=lambda x: x['k'])
        
        k_values = [d['k'] for d in data]
        valid_counts = [d['valid'] for d in data]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, valid_counts, 'o-', linewidth=2, markersize=8, color='green')
        
        # Add annotations for valid counts
        for k, valid in zip(k_values, valid_counts):
            plt.annotate(f'{valid}', (k, valid), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.xlabel('k (Number of examples)')
        plt.ylabel('Number of Valid Samples')
        plt.title(f'Number of Valid Samples vs k for {user}')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.ylim(0, max(valid_counts) * 1.1 if max(valid_counts) > 0 else 10)
        
        # Save the plot
        output_file = f"{output_dir}/{user}_valid.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved valid plot for {user}: {output_file}")

def print_valid_summary_stats(user_data):
    """Print summary statistics for valid samples"""
    print("\n" + "="*60)
    print("VALID SAMPLES SUMMARY STATISTICS")
    print("="*60)
    
    for user, data in user_data.items():
        data.sort(key=lambda x: x['k'])
        
        print(f"\n{user}:")
        print("-" * 20)
        for d in data:
            print(f"  k={d['k']:2d}: valid={d['valid']:3d}, acc={d['acc']:.3f}, a_rate={d['a_rate']:.3f}")
        
        # Find k with most valid samples
        max_valid_data = max(data, key=lambda x: x['valid'])
        print(f"  Most valid samples: k={max_valid_data['k']} ({max_valid_data['valid']} samples)")

if __name__ == "__main__":
    # Load data
    jsonl_file = "group1.jsonl"
    user_data = load_data(jsonl_file)
    
    print(f"Loaded data for {len(user_data)} users: {list(user_data.keys())}")
    
    # Create individual valid plots for each user
    plot_individual_valid_plots(user_data)
    
    # Create subplots for all users
    plot_valid_vs_k_subplots(user_data)
    
    # Create comparison plot
    plot_valid_vs_k_all_users(user_data)
    
    # Print summary statistics
    print_valid_summary_stats(user_data)
    
    print(f"\nAll valid plots saved in 'plots/' directory")
