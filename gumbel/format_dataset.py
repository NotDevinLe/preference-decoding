import json
import pickle
from datasets import load_dataset
from collections import defaultdict

# Load the PERSONA dataset
ds = load_dataset("SynthLabsAI/PERSONA")

def format_persona_dataset(dataset_split="train"):
    """
    Format PERSONA dataset into user-keyed structure.
    Each unique persona becomes a user with their prompt-response pairs.
    Creates both training and validation splits.
    """
    data_split = ds[dataset_split]
    
    # Group by persona (system prompt) - collect all preference pairs first
    user_preference_pairs = defaultdict(list)
    persona_to_user_id = {}
    user_counter = 0
    
    print(f"Processing {len(data_split)} examples...")
    
    for example in data_split:
        persona = example['persona']
        instruction = example['instruction']
        preferred_response = example['data']  # Preferred response
        unpreferred_response = example['original']  # Unpreferred response
        
        # Create user ID if first time seeing this persona
        if persona not in persona_to_user_id:
            persona_to_user_id[persona] = f"user_{user_counter}"
            user_counter += 1
        
        user_id = persona_to_user_id[persona]
        
        # Store as preference pair (we'll split later)
        user_preference_pairs[user_id].append({
            "prompt": instruction,
            "preferred": preferred_response,
            "unpreferred": unpreferred_response
        })
    
    print(f"Found {len(user_preference_pairs)} users")
    
    # Check if each user has exactly 200 preference pairs
    for user_id, pairs in user_preference_pairs.items():
        if len(pairs) != 200:
            print(f"Warning: {user_id} has {len(pairs)} preference pairs, expected 200")
    
    # Split into train (150 pairs = 300 outputs) and val (50 pairs = 100 outputs)
    train_data = {}
    val_data = {}
    
    for user_id, pairs in user_preference_pairs.items():
        # Shuffle pairs for random split
        import random
        random.seed(42)  # For reproducible splits
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Split: first 150 for train, last 50 for val
        train_pairs = shuffled_pairs[:150]
        val_pairs = shuffled_pairs[150:200]
        
        # Convert pairs to individual outputs for training set
        train_outputs = []
        for pair in train_pairs:
            train_outputs.append({
                "prompt": pair["prompt"],
                "output": pair["preferred"],
                "preference_label": "preferred"
            })
            train_outputs.append({
                "prompt": pair["prompt"],
                "output": pair["unpreferred"],
                "preference_label": "unpreferred"
            })
        
        # Convert pairs to individual outputs for validation set
        val_outputs = []
        for pair in val_pairs:
            val_outputs.append({
                "prompt": pair["prompt"],
                "output": pair["preferred"],
                "preference_label": "preferred"
            })
            val_outputs.append({
                "prompt": pair["prompt"],
                "output": pair["unpreferred"],
                "preference_label": "unpreferred"
            })
        
        train_data[user_id] = train_outputs
        val_data[user_id] = val_outputs
    
    print(f"Created training set: {len(train_data)} users, avg {sum(len(outputs) for outputs in train_data.values()) / len(train_data):.0f} outputs per user")
    print(f"Created validation set: {len(val_data)} users, avg {sum(len(outputs) for outputs in val_data.values()) / len(val_data):.0f} outputs per user")
    
    return train_data, val_data, persona_to_user_id

# Format the dataset
train_dataset, val_dataset, persona_mapping = format_persona_dataset("train")

# Save training and validation sets separately
print("Saving training and validation datasets...")

# 1. Training set - Pickle format (fastest loading)
with open("persona_train_dataset.pkl", "wb") as f:
    pickle.dump({
        "user_data": train_dataset,
        "persona_mapping": persona_mapping,
        "split": "train"
    }, f)
print("Saved training set as persona_train_dataset.pkl")

# 2. Validation set - Pickle format (fastest loading)
with open("persona_val_dataset.pkl", "wb") as f:
    pickle.dump({
        "user_data": val_dataset,
        "persona_mapping": persona_mapping,
        "split": "validation"
    }, f)
print("Saved validation set as persona_val_dataset.pkl")

# 3. Training set - JSON format (human readable)
with open("persona_train_dataset.json", "w") as f:
    json.dump({
        "user_data": train_dataset,
        "persona_mapping": persona_mapping,
        "split": "train"
    }, f, indent=2)
print("Saved training set as persona_train_dataset.json")

# 4. Validation set - JSON format (human readable)
with open("persona_val_dataset.json", "w") as f:
    json.dump({
        "user_data": val_dataset,
        "persona_mapping": persona_mapping,
        "split": "validation"
    }, f, indent=2)
print("Saved validation set as persona_val_dataset.json")

# Print comprehensive statistics
print("\nDataset Statistics:")
print(f"Total users (unique personas): {len(train_dataset)}")

# Training set stats
total_train_responses = sum(len(responses) for responses in train_dataset.values())
print(f"\nTraining Set:")
print(f"  Total responses: {total_train_responses}")
print(f"  Average responses per user: {total_train_responses / len(train_dataset):.0f}")
print(f"  Expected: 300 responses per user (150 preference pairs × 2)")

# Validation set stats  
total_val_responses = sum(len(responses) for responses in val_dataset.values())
print(f"\nValidation Set:")
print(f"  Total responses: {total_val_responses}")
print(f"  Average responses per user: {total_val_responses / len(val_dataset):.0f}")
print(f"  Expected: 100 responses per user (50 preference pairs × 2)")

# Show example user data
example_user = list(train_dataset.keys())[0]
print(f"\nExample user ({example_user}):")
print(f"  Persona: {list(persona_mapping.keys())[0][:100]}...")
print(f"  Training responses: {len(train_dataset[example_user])}")
print(f"  Validation responses: {len(val_dataset[example_user])}")
print(f"  Sample training response: {train_dataset[example_user][0]['output'][:100]}...")

print("\nDataset formatting complete!")
print("\nUsage:")
print("  Training: python main.py --dataset-path persona_train_dataset.pkl")
print("  Validation: python main.py --dataset-path persona_val_dataset.pkl")
