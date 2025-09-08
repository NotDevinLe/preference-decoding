import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Import the attribute prompts data to get the selection indices
import sys
sys.path.append('..')
from attribute_prompts import persona_selected

def load_data(jsonl_file):
    """Load data from JSONL file and group by user"""
    user_data = defaultdict(list)
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                user = data['user']
                user_data[user].append({
                    'lambda_val': data['lambda_val'],
                    'acc': data['acc'],
                    'system_prompt_list': data['system_prompt_list']
                })
    
    return user_data

def get_num_prompts_for_lambda(lambda_val):
    """Get the number of attribute prompts selected for a given lambda value"""
    if lambda_val in persona_selected:
        return len(persona_selected[lambda_val])
    else:
        # For lambda=0, return total number of persona prompts (from the data structure)
        return 400  # This is the total number of persona prompts when lambda=0

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
        # Sort by lambda value
        data.sort(key=lambda x: x['lambda_val'])
        
        lambda_values = [d['lambda_val'] for d in data]
        accuracies = [d['acc'] for d in data]
        
        # Convert lambda values to number of prompts
        num_prompts = [get_num_prompts_for_lambda(lam) for lam in lambda_values]
        
        # Plot on subplot
        ax = axes_flat[i]
        ax.plot(num_prompts, accuracies, 'o-', linewidth=2, markersize=6, color='blue')
        
        # Add annotations for lambda values
        for num_p, acc, lam in zip(num_prompts, accuracies, lambda_values):
            ax.annotate(f'λ={lam}', (num_p, acc), textcoords="offset points", 
                       xytext=(0,8), ha='center', fontsize=7)
        
        ax.set_xlabel('Number of Attribute Prompts')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{user}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(num_prompts)
        ax.set_ylim(0, max(accuracies) * 1.1 if max(accuracies) > 0 else 0.1)
    
    # Hide unused subplots
    for i in range(n_users, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Accuracy vs Number of Attribute Prompts for All Users', fontsize=16, y=0.98)
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
        # Sort by lambda value
        data.sort(key=lambda x: x['lambda_val'])
        
        lambda_values = [d['lambda_val'] for d in data]
        accuracies = [d['acc'] for d in data]
        
        # Convert lambda values to number of prompts
        num_prompts = [get_num_prompts_for_lambda(lam) for lam in lambda_values]
        
        plt.plot(num_prompts, accuracies, 'o-', linewidth=2, markersize=6, 
                label=user, color=colors[i])
    
    plt.xlabel('Number of Attribute Prompts')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Attribute Prompts - All Users Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis ticks based on unique number of prompts
    all_num_prompts = set()
    for data in user_data.values():
        for d in data:
            all_num_prompts.add(get_num_prompts_for_lambda(d['lambda_val']))
    plt.xticks(sorted(all_num_prompts))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_file}")

def plot_individual_users(user_data, output_dir='plots'):
    """Create individual plots for each user"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for user, data in user_data.items():
        # Sort by lambda value
        data.sort(key=lambda x: x['lambda_val'])
        
        lambda_values = [d['lambda_val'] for d in data]
        accuracies = [d['acc'] for d in data]
        
        # Convert lambda values to number of prompts
        num_prompts = [get_num_prompts_for_lambda(lam) for lam in lambda_values]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(num_prompts, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
        
        # Add annotations for lambda values
        for num_p, acc, lam in zip(num_prompts, accuracies, lambda_values):
            plt.annotate(f'λ={lam}', (num_p, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        plt.xlabel('Number of Attribute Prompts')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Number of Attribute Prompts for {user}')
        plt.grid(True, alpha=0.3)
        plt.xticks(num_prompts)
        plt.ylim(0, max(accuracies) * 1.1 if max(accuracies) > 0 else 0.1)
        
        # Save the plot
        output_file = f"{output_dir}/{user}_accuracy.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot for {user}: {output_file}")

def print_summary_stats(user_data):
    """Print summary statistics for each user"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for user, data in user_data.items():
        data.sort(key=lambda x: x['lambda_val'])
        
        print(f"\n{user}:")
        print("-" * 30)
        for d in data:
            num_prompts = get_num_prompts_for_lambda(d['lambda_val'])
            print(f"  λ={d['lambda_val']:8.5f}: acc={d['acc']:.3f}, prompts={num_prompts:3d}, type={d['system_prompt_list']}")
        
        # Find best lambda
        best_lambda_data = max(data, key=lambda x: x['acc'])
        num_prompts = get_num_prompts_for_lambda(best_lambda_data['lambda_val'])
        print(f"  Best λ: {best_lambda_data['lambda_val']:.5f} (acc={best_lambda_data['acc']:.3f}, prompts={num_prompts})")

if __name__ == "__main__":
    # Load data
    jsonl_file = "group1.jsonl"
    user_data = load_data(jsonl_file)
    
    print(f"Loaded data for {len(user_data)} users: {list(user_data.keys())}")
    
    # Create individual plots for each user
    plot_individual_users(user_data)
    
    # Create subplots for all users in one figure
    plot_all_users_subplots(user_data)
    
    # Create comparison plot
    plot_all_users_comparison(user_data)
    
    # Print summary statistics
    print_summary_stats(user_data)
    
    print(f"\nAll plots saved in 'plots/' directory")
