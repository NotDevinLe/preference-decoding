from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict, Any
"""
Class for putting hugging face datasets into a format that can be used by the reward model.

    Key Functions:
        - format_dataset: Formats a dataset into a format that can be used by the reward model

"""

class DatasetFormatter:
    def __init__(self, dataset_path: str, split: str = "train"):
        """
        Initialize DatasetFormatter.
        
        Args:
            dataset_path: Path to HuggingFace dataset (e.g., "databricks/databricks-dolly-15k")
            split: Dataset split to load (default: "train")
        """
        self.dataset = load_dataset(dataset_path, split=split)

    def format_dataset(self, users_col: str, prompt_col: str, chosen_col: str, rejected_col: str, save_path: str):
        """
        Formats a dataset into a format that can be used by the reward model.

        Args:
            users_col: Column name with user_id/persona_prompt (any way to differentiate between users)
            prompt_col: Column name with the question/prompt
            chosen_col: Column name with the chosen answer
            rejected_col: Column name with the rejected answer
            save_path: Path to save the formatted dataset files

        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        user_data_dict = {}
        for item in self.dataset:
            user = item.get(users_col)
            if user is None:
                continue
                
            prompt = item.get(prompt_col, "")
            chosen = item.get(chosen_col, "")
            rejected = item.get(rejected_col, "")
            
            if not prompt or not chosen or not rejected:
                continue
            
            if user not in user_data_dict:
                user_data_dict[user] = []
            
            user_data_dict[user].append({
                'user_id': user,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
        
        metadata = {}
        
        for user_idx, (user, formatted_data) in enumerate(sorted(user_data_dict.items())[50:]):
            if user_idx == 300:
                break

            metadata[f'user{user_idx}'] = user
            if not formatted_data:
                print(f"Warning: No data found for user {user}, skipping...")
                continue
            
            train, temp = train_test_split(formatted_data, test_size=0.4, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)
            
            with open(save_path / f'user{user_idx}_train.json', 'w') as f:
                json.dump(train, f, indent=2)
            with open(save_path / f'user{user_idx}_val.json', 'w') as f:
                json.dump(val, f, indent=2)
            with open(save_path / f'user{user_idx}_test.json', 'w') as f:
                json.dump(test, f, indent=2)
            
            print(f"Processed user {user} (index {user_idx}): {len(train)} train, {len(val)} val, {len(test)} test samples")

        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
def main():
    """
    Main function to format a dataset into a format that can be used by the reward model.
    """
    dataset_path = "SynthLabsAI/PERSONA"
    users_col = "persona"
    prompt_col = "instruction"
    chosen_col = "data"
    rejected_col = "original"
    save_path = "evals/persona"
    dataset_formatter = DatasetFormatter(dataset_path)
    dataset_formatter.format_dataset(users_col, prompt_col, chosen_col, rejected_col, save_path)

if __name__ == "__main__":
    main()