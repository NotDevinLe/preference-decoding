"""
YAML Configuration Loader for Sparse Autoencoder Training
Replaces the hardcoded Config class with YAML-based configuration.
"""

import yaml
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Configuration class that matches the original Config but loads from YAML."""
    
    # Model architecture
    n_dirs: int = 32768
    bs: int = 131072
    d_model: int = 768
    k: int = 32
    auxk: int = 256
    
    # Training hyperparameters
    lr: float = 1e-4
    eps: float = 6.25e-10
    clip_grad: Optional[float] = None
    auxk_coef: float = 1 / 32
    dead_toks_threshold: int = 10_000_000
    ema_multiplier: Optional[float] = None
    
    # Logging
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None


def load_config(config_path: str = "sparse.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return Config()
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract values from nested YAML structure
    model = config_dict.get('model', {})
    training = config_dict.get('training', {})
    logging = config_dict.get('logging', {})
    
    return Config(
        # Model
        n_dirs=model.get('n_dirs', 32768),
        bs=training.get('batch_size', 131072),
        d_model=model.get('d_model', 768),
        k=model.get('k', 32),
        auxk=model.get('auxk', 256),
        dead_toks_threshold=model.get('dead_toks_threshold', 10_000_000),
        
        # Training
        lr=training.get('learning_rate', 1e-4),
        eps=training.get('eps', 6.25e-10),
        clip_grad=training.get('clip_grad'),
        auxk_coef=training.get('auxk_coef', 1/32),
        ema_multiplier=training.get('ema_multiplier'),
        
        # Logging
        wandb_project=logging.get('wandb_project'),
        wandb_name=logging.get('wandb_name'),
    )


def save_config(config: Config, output_path: str = "config_export.yaml"):
    """
    Save a Config object to YAML file.
    
    Args:
        config: Config object to save
        output_path: Path to save the YAML file
    """
    config_dict = {
        'model': {
            'n_dirs': config.n_dirs,
            'd_model': config.d_model,
            'k': config.k,
            'auxk': config.auxk,
            'dead_toks_threshold': config.dead_toks_threshold,
        },
        'training': {
            'batch_size': config.bs,
            'learning_rate': config.lr,
            'eps': config.eps,
            'clip_grad': config.clip_grad,
            'auxk_coef': config.auxk_coef,
            'ema_multiplier': config.ema_multiplier,
        },
        'logging': {
            'wandb_project': config.wandb_project,
            'wandb_name': config.wandb_name,
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Example usage
    config = load_config("sparse.yaml")
    print("Loaded configuration:")
    print(f"Model dimensions: {config.d_model}")
    print(f"Dictionary size: {config.n_dirs}")
    print(f"Sparsity level: {config.k}")
    print(f"Learning rate: {config.lr}")
    print(f"Batch size: {config.bs}")
