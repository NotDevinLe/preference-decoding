import torch
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class UserSample:
    """Single user sample containing prompt and output"""
    user_id: str
    prompt: str
    output: str
    features: Optional[torch.Tensor] = None  # Optional pre-computed features

@dataclass
class BatchSample:
    """Batch of user samples"""
    user_samples: List[UserSample]
    X: torch.Tensor  # [batch_size, d] feature matrix
    user_data: Dict[str, Any]  # Data for reward scoring

class DataSampler:
    """
    Sample users and their data for training the sparse attribute model.
    This replaces the dummy sample_X function in collector.py.
    """
    
    def __init__(self, 
                 dataset_path: Optional[str] = None,
                 num_attributes: int = 100,
                 feature_dim: int = 100):
        """
        Initialize the data sampler.
        
        Args:
            dataset_path: Path to your preference dataset (e.g., NPZ file)
            num_attributes: Number of attributes/features (d)
            feature_dim: Dimension of feature embeddings
        """
        self.dataset_path = dataset_path
        self.num_attributes = num_attributes
        self.feature_dim = feature_dim
        
        # In practice, you'd load your actual dataset here
        self.users_data = self._load_dataset()
        
        logging.info(f"DataSampler initialized with {len(self.users_data)} users")
        logging.info(f"Attributes: {num_attributes}, Feature dim: {feature_dim}")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """
        Load your actual preference dataset.
        Supports multiple formats: .pkl, .json, .npz
        """
        if self.dataset_path:
            try:
                dataset_path = self.dataset_path
                
                # Handle different file formats
                if dataset_path.endswith('.pkl'):
                    import pickle
                    with open(dataset_path, 'rb') as f:
                        data = pickle.load(f)
                    logging.info(f"Loaded pickle dataset from {dataset_path}")
                    
                    # Handle PERSONA dataset format
                    if 'user_data' in data:
                        return data['user_data']
                    else:
                        return data
                        
                elif dataset_path.endswith('.json'):
                    with open(dataset_path, 'r') as f:
                        data = json.load(f)
                    logging.info(f"Loaded JSON dataset from {dataset_path}")
                    
                    # Handle PERSONA dataset format
                    if 'user_data' in data:
                        return data['user_data']
                    else:
                        return data
                        
                elif dataset_path.endswith('.npz'):
                    # Legacy support for NPZ files
                    data = np.load(dataset_path, allow_pickle=True)
                    logging.info(f"Loaded NPZ dataset from {dataset_path}")
                    return self._process_loaded_data(data)
                    
                else:
                    logging.warning(f"Unsupported file format: {dataset_path}")
                    
            except Exception as e:
                logging.warning(f"Failed to load dataset from {self.dataset_path}: {e}")
                logging.info("Falling back to dummy data generation")
        
        # Fallback: generate dummy data
        return self._generate_dummy_data()
    
    def _process_loaded_data(self, data) -> Dict[str, Any]:
        """
        Process loaded dataset into the format needed for sampling.
        This is highly dataset-specific and should be customized.
        """
        # Example processing for a typical preference dataset
        # Adapt this based on your actual data structure
        
        users_data = {}
        
        # If your data has user IDs, prompts, and outputs
        if 'user_ids' in data and 'prompts' in data and 'outputs' in data:
            user_ids = data['user_ids']
            prompts = data['prompts'] 
            outputs = data['outputs']
            
            for user_id, prompt, output in zip(user_ids, prompts, outputs):
                if user_id not in users_data:
                    users_data[user_id] = []
                users_data[user_id].append({
                    'prompt': prompt,
                    'output': output
                })
        
        # If you have reward matrix data like Y_chosen
        elif 'Y_chosen' in data:
            # Convert reward matrix to user-prompt-output format
            Y = data['Y_chosen']  # Shape: [num_samples, num_attributes]
            
            # Create synthetic users and prompts from the reward data
            for i in range(min(1000, Y.shape[0])):  # Limit for testing
                user_id = f"user_{i}"
                users_data[user_id] = [{
                    'prompt': f"Sample prompt {i}",
                    'output': f"Sample output {i}",
                    'features': Y[i]  # Use reward values as features
                }]
        
        else:
            logging.warning("Unrecognized data format, generating dummy data")
            return self._generate_dummy_data()
        
        return users_data
    
    def _generate_dummy_data(self) -> Dict[str, Any]:
        """Generate dummy data for testing"""
        users_data = {}
        
        for user_id in range(100):  # 100 dummy users
            user_key = f"user_{user_id}"
            users_data[user_key] = []
            
            # Each user has 5-10 prompt-output pairs
            num_samples = random.randint(5, 10)
            for sample_id in range(num_samples):
                users_data[user_key].append({
                    'prompt': f"User {user_id} asks about topic {sample_id}",
                    'output': f"Response from user {user_id} about topic {sample_id}",
                })
        
        return users_data
    
    def sample_batch(self, batch_size: int, device: str = "cpu") -> BatchSample:
        """
        Sample a batch of users and their data.
        
        Args:
            batch_size: Number of users to sample
            device: Device to place tensors on
            
        Returns:
            BatchSample containing user samples and feature matrix
        """
        device_t = torch.device(device)
        user_samples = []
        features_list = []
        
        # Sample users uniformly
        available_users = list(self.users_data.keys())
        if len(available_users) == 0:
            raise ValueError("No users available in dataset")
        
        for _ in range(batch_size):
            # Sample user
            user_id = random.choice(available_users)
            user_entries = self.users_data[user_id]
            
            # Sample one prompt-output pair from this user
            entry = random.choice(user_entries)
            
            # Create user sample
            user_sample = UserSample(
                user_id=user_id,
                prompt=entry['prompt'],
                output=entry['output'],
                features=entry.get('features', None)
            )
            user_samples.append(user_sample)
            
            # Generate or extract features for this user
            if 'features' in entry and entry['features'] is not None:
                # Use pre-computed features
                features = torch.tensor(entry['features'], dtype=torch.float32)
                # Ensure correct dimensionality
                if len(features.shape) == 1 and features.shape[0] == self.num_attributes:
                    features_list.append(features)
                else:
                    # Reshape or pad/truncate as needed
                    if features.shape[0] > self.num_attributes:
                        features = features[:self.num_attributes]
                    elif features.shape[0] < self.num_attributes:
                        padding = torch.zeros(self.num_attributes - features.shape[0])
                        features = torch.cat([features, padding])
                    features_list.append(features)
            else:
                # Generate random features
                features = torch.randn(self.num_attributes)
                features_list.append(features)
        
        # Stack features into batch matrix
        X = torch.stack(features_list).to(device_t)  # [batch_size, d]
        
        # Create user_data dict for reward scoring
        user_data = {
            'prompts': [sample.prompt for sample in user_samples],
            'outputs': [sample.output for sample in user_samples],
            'user_ids': [sample.user_id for sample in user_samples]
        }
        
        return BatchSample(
            user_samples=user_samples,
            X=X,
            user_data=user_data
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset"""
        total_samples = sum(len(entries) for entries in self.users_data.values())
        
        return {
            'num_users': len(self.users_data),
            'total_samples': total_samples,
            'avg_samples_per_user': total_samples / max(len(self.users_data), 1),
            'num_attributes': self.num_attributes,
            'feature_dim': self.feature_dim
        }


# Utility function to create sampler from config
def create_data_sampler(config: Dict[str, Any]) -> DataSampler:
    """Create data sampler from configuration dict"""
    return DataSampler(
        dataset_path=config.get('dataset_path', None),
        num_attributes=config.get('num_attributes', 100),
        feature_dim=config.get('feature_dim', 100)
    )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy data
    sampler = DataSampler(num_attributes=10, feature_dim=10)
    
    # Print dataset stats
    stats = sampler.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Sample a batch
    batch = sampler.sample_batch(batch_size=4)
    
    print(f"\nSampled batch:")
    print(f"  Batch size: {len(batch.user_samples)}")
    print(f"  Feature matrix shape: {batch.X.shape}")
    print(f"  User IDs: {batch.user_data['user_ids']}")
    print(f"  Sample prompts: {batch.user_data['prompts'][:2]}...")  # Show first 2
    
    # Show feature statistics
    print(f"  Feature statistics:")
    print(f"    Mean: {batch.X.mean().item():.3f}")
    print(f"    Std: {batch.X.std().item():.3f}")
    print(f"    Shape: {batch.X.shape}")