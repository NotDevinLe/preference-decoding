from datasets import load_dataset
import random
import json
import os

ds = load_dataset("SynthLabsAI/PERSONA")

distinct_personas = set()
for example in ds["train"]:
    distinct_personas.add(example["persona"])

random.seed(0)

chosen_personas = random.sample(list(distinct_personas), 90)

data = {}

for persona in chosen_personas:
    data[persona] = []

for example in ds["train"]:
    if example["persona"] in chosen_personas:
        data[example["persona"]].append({"prompt": example["instruction"], "chosen": example["data"], "rejected": example["original"]})

# Load existing metadata or create new structure
metadata_path = "user_metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # Find the next user ID to start from
    existing_users = [user["user_id"] for user in metadata["users"]]
    if existing_users:
        last_user_num = max([int(uid.replace("user", "")) for uid in existing_users])
        curr = last_user_num + 1
    else:
        curr = 21

# Process each persona and create user files + metadata
users_added = 0
total_training_rows = metadata["dataset_info"]["total_training_rows"]
total_test_rows = metadata["dataset_info"]["total_test_rows"]

for persona, examples in data.items():
    user_id = f"user{curr}"
    
    # Create user data files
    with open(f"{user_id}_train.json", "w") as f:
        json.dump(examples[:150], f)
    with open(f"{user_id}_test.json", "w") as f:
        json.dump(examples[150:], f)
    
    # Create user metadata entry
    user_metadata = {
        "user_id": user_id,
        "persona_text": persona,
        "total_available_rows": len(examples),
        "training_rows": min(150, len(examples)),
        "test_rows": min(50, max(0, len(examples) - 150)),
        "persona_preview": persona[:100] + "..." if len(persona) > 100 else persona
    }
    
    # Add to metadata
    metadata["users"].append(user_metadata)
    
    # Update totals
    users_added += 1
    total_training_rows += user_metadata["training_rows"]
    total_test_rows += user_metadata["test_rows"]
    
    curr += 1

# Update dataset info
metadata["dataset_info"]["total_users"] += users_added
metadata["dataset_info"]["total_training_rows"] = total_training_rows
metadata["dataset_info"]["total_test_rows"] = total_test_rows

# Save updated metadata
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Added {users_added} new users to the dataset")
print(f"Total users: {metadata['dataset_info']['total_users']}")
print(f"Total training rows: {metadata['dataset_info']['total_training_rows']}")
print(f"Total test rows: {metadata['dataset_info']['total_test_rows']}")