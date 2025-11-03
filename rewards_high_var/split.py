import torch
import os

def split_and_label_tensors(input_dir, output_dir, column_counts):
    """
    Split tensor files and create labeled versions with different numbers of columns.
    
    Args:
        input_dir: Directory containing the original tensor files
        output_dir: Directory to save the split files
        column_counts: List of column counts to create (e.g., [10, 20, 50, 100])
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original tensor files
    train_file = os.path.join(input_dir, "rewards_persona_testing_train.pt")
    val_file = os.path.join(input_dir, "rewards_persona_testing_val.pt")
    
    print(f"Loading tensors from {input_dir}...")
    train_tensor = torch.load(train_file)
    val_tensor = torch.load(val_file)
    
    print(f"Original train tensor shape: {train_tensor.shape}")
    print(f"Original val tensor shape: {val_tensor.shape}")
    
    # Get the maximum number of columns available
    max_cols = min(train_tensor.shape[1], val_tensor.shape[1])
    print(f"Maximum columns available: {max_cols}")
    
    # Create split versions for each column count
    for n_cols in column_counts:
        if n_cols > max_cols:
            print(f"Warning: Requested {n_cols} columns but only {max_cols} available. Skipping.")
            continue
            
        print(f"Creating splits with {n_cols} columns...")
        
        # Take the first n_cols columns
        train_split = train_tensor[:, :n_cols]
        val_split = val_tensor[:, :n_cols]
        
        # Create output filenames
        train_output = os.path.join(output_dir, f"train_o{n_cols}.pt")
        val_output = os.path.join(output_dir, f"val_o{n_cols}.pt")
        
        # Save the split tensors
        torch.save(train_split, train_output)
        torch.save(val_split, val_output)
        
        print(f"Saved {train_output} with shape {train_split.shape}")
        print(f"Saved {val_output} with shape {val_split.shape}")

if __name__ == "__main__":
    # Define the input and output directories
    input_dir = "/gscratch/ark/devinl6/preference/preference-decoding/rewards_high_var"
    output_dir = "/gscratch/ark/devinl6/preference/preference-decoding/rewards_high_var"
    
    # Define the column counts you want to create
    # You can modify this list to include the specific numbers you want
    column_counts = [10, 20, 50, 100, 150, 200]
    
    print("Starting tensor splitting and labeling...")
    split_and_label_tensors(input_dir, output_dir, column_counts)
    print("Done!")