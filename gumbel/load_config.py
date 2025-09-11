#!/usr/bin/env python3
"""
Utility to load and validate configuration from config.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

def auto_detect_num_attributes(attribute_prompts_path: str) -> int:
    """Automatically detect number of attributes from prompts file"""
    try:
        prompts_file = Path(attribute_prompts_path)
        if not prompts_file.exists():
            print(f"Warning: Attribute prompts file not found: {attribute_prompts_path}")
            print("Using default d=100. Create the prompts file or set d manually in config.")
            return 100
            
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            num_attributes = len(data)
        elif isinstance(data, dict) and 'prompts' in data:
            num_attributes = len(data['prompts'])
        elif isinstance(data, dict):
            # Assume each key is an attribute
            num_attributes = len(data)
        else:
            raise ValueError(f"Unsupported attribute prompts format in {attribute_prompts_path}")
        
        print(f"Auto-detected {num_attributes} attributes from {attribute_prompts_path}")
        return num_attributes
        
    except Exception as e:
        print(f"Error auto-detecting attributes: {e}")
        print("Using default d=100")
        return 100

def resolve_auto_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve 'auto' values in config"""
    config = config.copy()  # Don't modify original
    
    # Auto-detect number of attributes if set to "auto"
    if config['model'].get('d') == 'auto':
        attribute_prompts_path = config['data']['attribute_prompts_path']
        config['model']['d'] = auto_detect_num_attributes(attribute_prompts_path)
    
    return config

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file and resolve auto values"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Resolve any "auto" values
    config = resolve_auto_values(config)
    
    return config

def get_collector_args(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract collector arguments from config"""
    args = {
        'd': config['model']['d'],
        'dataset_path': config['data']['dataset_path'],
        'attribute_prompts_path': config['data']['attribute_prompts_path'],
        'vllm_model': config['vllm']['model_name'],
        'gpu_memory_util': config['vllm']['gpu_memory_util'],
        'host': config['servers']['collector']['host'],
        'port': config['servers']['collector']['port'],
        'device': config['servers']['collector']['device'],
        'log_level': config['monitoring']['log_level']
    }
    
    if overrides:
        args.update(overrides)
    
    return args

def get_learner_args(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract learner arguments from config"""
    args = {
        'd': config['model']['d'],
        'k': config['model']['k'],
        'lr': config['model']['lr'],
        'sparsity_weight': config['model']['sparsity_weight'],
        'tau_init': config['model']['tau_init'],
        'host': config['servers']['learner']['host'],
        'port': config['servers']['learner']['port'],
        'device': config['servers']['learner']['device'],
        'checkpoint_dir': config['servers']['learner']['checkpoint_dir'],
        'use_wandb': config['servers']['learner']['use_wandb'],
        'log_level': config['monitoring']['log_level']
    }
    
    if overrides:
        args.update(overrides)
    
    return args

def get_coordinator_args(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract coordinator arguments from config"""
    args = {
        'max_steps': config['training']['max_steps'],
        'log_freq': config['training']['log_freq'],
        'checkpoint_freq': config['training']['checkpoint_freq'],
        'users_per_batch': config['training']['users_per_batch'],
        'samples_per_user': config['training']['samples_per_user'],
        'queue_size': config['training']['queue_size'],
        'replay_buffer_size': config['training']['replay_buffer_size'],
        'replay_ratio': config['training']['replay_ratio'],
        'enable_monitoring': config['monitoring']['enable_monitoring'],
        'enable_wandb': config['monitoring']['enable_wandb'],
        'plot_update_interval': config['monitoring']['plot_update_interval'],
        'collector_url': f"http://{config['servers']['collector']['host']}:{config['servers']['collector']['port']}",
        'learner_url': f"http://{config['servers']['learner']['host']}:{config['servers']['learner']['port']}",
        'timeouts': config['timeouts']
    }
    
    if overrides:
        args.update(overrides)
    
    return args

def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the current configuration"""
    print("=== Configuration Summary ===")
    print(f"Model: {config['model']['d']} attributes -> {config['model']['k']} components")
    print(f"Data: {config['data']['dataset_path']}")
    print(f"Attribute Prompts: {config['data']['attribute_prompts_path']}")
    print(f"VLLM Model: {config['vllm']['model_name']}")
    print(f"Collector: {config['servers']['collector']['host']}:{config['servers']['collector']['port']} ({config['servers']['collector']['device']})")
    print(f"Learner: {config['servers']['learner']['host']}:{config['servers']['learner']['port']} ({config['servers']['learner']['device']})")
    print(f"Training: {config['training']['max_steps']} steps")
    print(f"Monitoring: {'Enabled' if config['monitoring']['enable_monitoring'] else 'Disabled'}")
    print("===============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display configuration")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--component", type=str, choices=['collector', 'learner', 'coordinator'], 
                       help="Show args for specific component")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        
        if args.component:
            if args.component == 'collector':
                collector_args = get_collector_args(config)
                print("Collector arguments:")
                for key, value in collector_args.items():
                    print(f"  {key}: {value}")
            elif args.component == 'learner':
                learner_args = get_learner_args(config)
                print("Learner arguments:")
                for key, value in learner_args.items():
                    print(f"  {key}: {value}")
            elif args.component == 'coordinator':
                coordinator_args = get_coordinator_args(config)
                print("Coordinator arguments:")
                for key, value in coordinator_args.items():
                    print(f"  {key}: {value}")
        else:
            print_config_summary(config)
            
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)