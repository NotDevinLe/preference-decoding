import json
import os
import glob
import random

def split_dataset(input_file, output_test_file, num_train_samples=80):
    """
    Split a dataset file by keeping the first num_train_samples in the original file
    and moving the rest to a new test file.
    
    Args:
        input_file: Path to the original train JSON file
        output_test_file: Path to the new test JSON file
        num_train_samples: Number of samples to keep in the train file (default: 80)
    """
    # Read the original file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {input_file}: {len(data)} total samples")
    
    # Split the data
    train_data = data[:num_train_samples]
    test_data = data[num_train_samples:]
    
    # Write the train file with only the first num_train_samples
    with open(input_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"  - Kept {len(train_data)} samples in {input_file}")
    
    # Write the test file with the remaining samples
    with open(output_test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"  - Created {output_test_file} with {len(test_data)} samples")

def create_validation_set(test_dir="."):
    """
    Create a validation set by pairing each test question from one user with
    a randomly sampled answer from another user for the same question.
    
    Args:
        test_dir: Directory containing the test JSON files
    """
    # Find all test files
    pattern = "user*_test.json"
    test_files = glob.glob(os.path.join(test_dir, pattern))
    test_files.sort()
    
    if len(test_files) == 0:
        print("No test files found")
        return
    
    print(f"Found {len(test_files)} test files to process\n")
    
    # Load all test data
    all_test_data = {}
    for test_file in test_files:
        user_id = os.path.basename(test_file).replace('_test.json', '')
        with open(test_file, 'r') as f:
            all_test_data[user_id] = json.load(f)
    
    # Create validation set for each user
    for user_id in all_test_data.keys():
        user_test_data = all_test_data[user_id]
        validation_data = []
        
        for question_idx, question_data in enumerate(user_test_data):
            # Get the prompt and chosen answer from current user
            prompt = question_data['prompt']
            chosen = question_data['chosen']
            
            # Randomly sample another user for the rejected answer
            other_users = [uid for uid in all_test_data.keys() if uid != user_id]
            sampled_user = random.choice(other_users)
            
            # Get the rejected answer from the sampled user for the same question
            if question_idx < len(all_test_data[sampled_user]):
                rejected = all_test_data[sampled_user][question_idx]['chosen']
                
                validation_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        # Write validation file
        validation_file = os.path.join(test_dir, f"{user_id}_val.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"Created {validation_file} with {len(validation_data)} samples")

def create_modified_train(train_dir="."):
    """
    Create a modified train set by pairing each train question from one user with
    a randomly sampled answer from another user for the same question.
    
    Args:
        train_dir: Directory containing the train JSON files
    """
    # Find all train files
    pattern = "user*_train.json"
    train_files = glob.glob(os.path.join(train_dir, pattern))
    train_files.sort()
    
    if len(train_files) == 0:
        print("No train files found")
        return
    
    print(f"Found {len(train_files)} train files to process\n")
    
    # Load all train data
    all_train_data = {}
    for train_file in train_files:
        user_id = os.path.basename(train_file).replace('_train.json', '')
        with open(train_file, 'r') as f:
            all_train_data[user_id] = json.load(f)
    
    # Create modified train set for each user
    for user_id in all_train_data.keys():
        user_train_data = all_train_data[user_id]
        modified_train_data = []
        
        for question_idx, question_data in enumerate(user_train_data):
            # Get the prompt and chosen answer from current user
            prompt = question_data['prompt']
            chosen = question_data['chosen']
            
            # Randomly sample another user for the rejected answer
            other_users = [uid for uid in all_train_data.keys() if uid != user_id]
            sampled_user = random.choice(other_users)
            
            # Get the rejected answer from the sampled user for the same question
            if question_idx < len(all_train_data[sampled_user]):
                rejected = all_train_data[sampled_user][question_idx]['chosen']
                
                modified_train_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        # Write modified train file
        modified_train_file = os.path.join(train_dir, f"{user_id}_modtrain.json")
        with open(modified_train_file, 'w') as f:
            json.dump(modified_train_data, f, indent=2)
        
        print(f"Created {modified_train_file} with {len(modified_train_data)} samples")

def main():
    # First, let's check if we need to create test files
    directory = "."
    train_pattern = "user*_train.json"
    test_pattern = "user*_test.json"
    
    train_files = glob.glob(os.path.join(directory, train_pattern))
    test_files = glob.glob(os.path.join(directory, test_pattern))
    
    # If no test files exist, create them by splitting train files
    if len(test_files) == 0 and len(train_files) > 0:
        print("No test files found. Creating test files by splitting train data...\n")
        for train_file in sorted(train_files):
            test_file = train_file.replace('_train.json', '_test.json')
            if not os.path.exists(test_file):
                split_dataset(train_file, test_file)
                print()
    
    # Create modified train set
    print("Creating modified train set...\n")
    create_modified_train()
    
    # Now create validation set (skip if already exists)
    val_pattern = "user*_val.json"
    val_files = glob.glob(os.path.join(directory, val_pattern))
    
    if len(val_files) == 0:
        print("\nCreating validation set...\n")
        create_validation_set()
    else:
        print(f"\nValidation set already exists ({len(val_files)} files found). Skipping.")

if __name__ == "__main__":
    main()

