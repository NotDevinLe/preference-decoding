from datasets import load_dataset
import json
import random
from collections import defaultdict

# Load the dataset
ds = load_dataset("SynthLabsAI/PERSONA")

print("Loading dataset and extracting personas...")

# Look for persona-related fields
persona_fields = ['input persona', 'persona', 'persona_id', 'persona_name', 'persona_type', 'role', 'character']
found_persona_field = None

for field in persona_fields:
    if field in ds['train'].features:
        found_persona_field = field
        break

if not found_persona_field:
    print("No persona field found. Available fields:")
    for field in ds['train'].features.keys():
        print(f"  - {field}")
    exit(1)

print(f"Found persona field: '{found_persona_field}'")

# Group data by persona
persona_data = defaultdict(list)
total_rows = len(ds['train'])

for i, row in enumerate(ds['train']):
    persona = row[found_persona_field]
    persona_data[persona].append({
        "row_index": i,
        "data": row
    })

print(f"Found {len(persona_data)} unique personas")
print(f"Total rows: {total_rows}")

# Get personas with sufficient data (at least 200 rows for 150+50 split)
personas_with_sufficient_data = {
    persona: data for persona, data in persona_data.items() 
    if len(data) >= 200
}

print(f"Personas with ≥200 rows: {len(personas_with_sufficient_data)}")

# Select 10 personas randomly from those with sufficient data
if len(personas_with_sufficient_data) >= 10:
    selected_personas = random.sample(list(personas_with_sufficient_data.keys()), 10)
else:
    print(f"Warning: Only {len(personas_with_sufficient_data)} personas have sufficient data")
    selected_personas = list(personas_with_sufficient_data.keys())

print(f"Selected {len(selected_personas)} personas for processing")

# Process each selected persona
training_data = []
test_data = []

for persona in selected_personas:
    data = personas_with_sufficient_data[persona]
    print(f"\nProcessing persona: {persona[:100]}...")
    print(f"  Total rows: {len(data)}")
    
    # Shuffle the data for this persona
    random.shuffle(data)
    
    # Split: first 150 for training, next 50 for test
    train_split = data[:150]
    test_split = data[150:200]
    
    print(f"  Training split: {len(train_split)} rows")
    print(f"  Test split: {len(test_split)} rows")
    
    # Convert to the required format for this persona
    persona_training_data = []
    persona_test_data = []
    
    for item in train_split:
        row = item["data"]
        # Look for the fields that match your format
        prompt = row.get('instruction', row.get('prompt', 'NO_PROMPT_FOUND'))
        chosen = row.get('data', 'NO_CHOSEN_FOUND')  # Correct response is in 'data' column
        rejected = row.get('original', 'NO_REJECTED_FOUND')  # Rejected response is in 'original' column
        
        persona_training_data.append({
            "chosen": chosen,
            "rejected": rejected,
            "prompt": prompt
        })
    
    for item in test_split:
        row = item["data"]
        prompt = row.get('instruction', row.get('prompt', 'NO_PROMPT_FOUND'))
        chosen = row.get('data', 'NO_CHOSEN_FOUND')  # Correct response is in 'data' column
        rejected = row.get('original', 'NO_REJECTED_FOUND')  # Rejected response is in 'original' column
        
        persona_test_data.append({
            "chosen": chosen,
            "rejected": rejected,
            "prompt": prompt
        })
    
    # Save individual persona files with simple user IDs
    persona_index = selected_personas.index(persona) + 11  # Start from user_11
    user_id = f"user_{persona_index}"
    
    # Save training data for this persona
    training_file = f"../data/persona_pref/{user_id}_train.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(persona_training_data, f, indent=2, ensure_ascii=False)
    
    # Save test data for this persona
    test_file = f"../data/persona_pref/{user_id}_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(persona_test_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved: {training_file}")
    print(f"  Saved: {test_file}")
    
    # Also add to combined datasets for summary
    training_data.extend(persona_training_data)
    test_data.extend(persona_test_data)

# Create metadata file with user information
user_metadata = []
for i, persona in enumerate(selected_personas):
    user_id = f"user_{i + 11}"
    data = personas_with_sufficient_data[persona]
    
    user_metadata.append({
        "user_id": user_id,
        "persona_text": persona,
        "total_available_rows": len(data),
        "training_rows": 150,
        "test_rows": 50,
        "persona_preview": persona[:200] + "..." if len(persona) > 200 else persona
    })

# Save metadata file
metadata_file = "../data/persona_pref/user_metadata.json"
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump({
        "dataset_info": {
            "source": "SynthLabsAI/PERSONA",
            "total_users": len(selected_personas),
            "total_training_rows": len(training_data),
            "total_test_rows": len(test_data),
            "rows_per_user": {
                "training": 150,
                "test": 50
            }
        },
        "users": user_metadata
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved user metadata to: {metadata_file}")

print(f"\nFinal dataset sizes:")
print(f"  Total training: {len(training_data)} rows")
print(f"  Total test: {len(test_data)} rows")

# Show sample of the format
print(f"\nSample training data format:")
if training_data:
    sample = training_data[0]
    print(f"  Prompt: {sample['prompt'][:100]}...")
    print(f"  Chosen: {sample['chosen'][:100]}...")
    print(f"  Rejected: {sample['rejected'][:100]}...")