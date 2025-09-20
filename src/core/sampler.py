import pickle
import random
import logging
from typing import Dict, Any, List

def load_user_data(dataset_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Load user data from pickle file"""
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle PERSONA dataset format
        if 'user_data' in data:
            return data['user_data']
        else:
            return data
            
    except Exception as e:
        logging.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise

def sample_batch(user_data: Dict[str, List[Dict[str, str]]], 
                users_per_batch: int, 
                samples_per_user: int) -> Dict[str, Any]:
    """
    Sample u users and n outputs per user.
    
    Args:
        user_data: Dict mapping user_id -> list of {"prompt": ..., "output": ...}
        users_per_batch: Number of users to sample (u)
        samples_per_user: Number of samples per user (n)
    
    Returns:
        Dict with 'prompts', 'outputs', 'user_ids'
        Each list has length users_per_batch * samples_per_user
    """
    available_users = list(user_data.keys())
    if len(available_users) == 0:
        raise ValueError("No users available in dataset")
    
    # Sample u users uniformly
    if len(available_users) < users_per_batch:
        # Sample with replacement if not enough users
        sampled_users = random.choices(available_users, k=users_per_batch)
    else:
        # Sample without replacement
        sampled_users = random.sample(available_users, users_per_batch)
    
    # Collect data from each user
    all_prompts = []
    all_outputs = []
    all_user_ids = []
    
    for user_id in sampled_users:
        user_entries = user_data[user_id]
        
        # Sample n entries from this user
        if len(user_entries) < samples_per_user:
            # Sample with replacement if user doesn't have enough entries
            sampled_entries = random.choices(user_entries, k=samples_per_user)
        else:
            # Sample without replacement
            sampled_entries = random.sample(user_entries, samples_per_user)
        
        # Add to collections
        all_prompts.extend([entry['prompt'] for entry in sampled_entries])
        all_outputs.extend([entry['output'] for entry in sampled_entries])
        all_user_ids.extend([user_id] * samples_per_user)
    
    return {
        'prompts': all_prompts,
        'outputs': all_outputs,
        'user_ids': all_user_ids
    }

class DataSampler:
    """Simple data sampler for user preference data"""
    
    def __init__(self, dataset_path: str = None, users_per_batch: int = 4, samples_per_user: int = 8):
        """
        Initialize the data sampler.
        
        Args:
            dataset_path: Path to the dataset pickle file
            users_per_batch: Default number of users to sample per batch
            samples_per_user: Default number of samples per user
        """
        self.users_per_batch = users_per_batch
        self.samples_per_user = samples_per_user
        
        if dataset_path:
            self.user_data = load_user_data(dataset_path)
            logging.info(f"Loaded {len(self.user_data)} users from {dataset_path}")
        else:
            logging.error("No dataset path provided")
            raise ValueError("Dataset path is required - cannot proceed without data")
    
    def __call__(self, users_per_batch: int = None, samples_per_user: int = None, device: str = "cpu") -> Dict[str, Any]:
        """
        Sample a batch by specifying users and samples per user.
        
        Args:
            users_per_batch: Number of users to sample (uses default if None)
            samples_per_user: Number of samples per user (uses default if None)
            device: Device (ignored, kept for backward compatibility)
        """
        _ = device  # Unused, kept for backward compatibility
        
        users = users_per_batch or self.users_per_batch
        samples = samples_per_user or self.samples_per_user
        
        return sample_batch(self.user_data, users, samples)
    
    def get_stats(self) -> Dict[str, float]:
        """Get dataset statistics"""
        return {
            'num_users': len(self.user_data),
            'total_samples': sum(len(entries) for entries in self.user_data.values()),
            'avg_samples_per_user': sum(len(entries) for entries in self.user_data.values()) / len(self.user_data)
        }