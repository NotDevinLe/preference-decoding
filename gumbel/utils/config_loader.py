#!/usr/bin/env python3
"""
Configuration loader for YAML and JSON config files.
Supports both formats and provides validation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigLoader:
    """Loads and validates experiment configuration files."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML not installed. Run: pip install pyyaml")
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # Try to detect format from content
                content = f.read()
                f.seek(0)
                if content.strip().startswith('{'):
                    config = json.load(f)
                else:
                    if not YAML_AVAILABLE:
                        raise ImportError("PyYAML not installed. Run: pip install pyyaml")
                    config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def resolve_auto_d(config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve 'd: auto' by counting attributes in prompts file."""
        if config.get("model", {}).get("d") == "auto":
            prompts_path = config.get("data", {}).get("attribute_prompts_path")
            if not prompts_path:
                raise ValueError("Cannot resolve 'd: auto' - no attribute_prompts_path specified")
            
            prompts_path = Path(prompts_path)
            if not prompts_path.exists():
                raise FileNotFoundError(f"Attribute prompts file not found: {prompts_path}")
            
            with open(prompts_path, 'r') as f:
                prompts_data = json.load(f)
            
            if isinstance(prompts_data, list):
                d = len(prompts_data)
            elif isinstance(prompts_data, dict) and "prompts" in prompts_data:
                d = len(prompts_data["prompts"])
            else:
                raise ValueError("Invalid attribute prompts file format")
            
            config["model"]["d"] = d
            print(f"Auto-detected d={d} from {prompts_path}")
        
        return config
    
    @staticmethod
    def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides from config."""
        env_vars = config.get("environment", {})
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        return config
    
    @staticmethod
    def get_collector_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract collector server configuration."""
        return {
            "d": config["model"]["d"],
            "dataset_path": config["data"]["dataset_path"],
            "model_name": config["vllm"]["model_name"],
            "vllm_server_url": config["vllm"]["server_url"],
            "attribute_prompts_path": config["data"]["attribute_prompts_path"],
            "host": config["servers"]["collector"]["host"],
            "port": config["servers"]["collector"]["port"],
            "device": config["servers"]["collector"]["device"],
            "log_level": config["monitoring"]["log_level"],
            "concurrency": config["servers"]["collector"].get("concurrency", 256),
        }
    
    @staticmethod
    def get_learner_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learner server configuration."""
        return {
            "d": config["model"]["d"],
            "k": config["model"]["k"],
            "lr": config["model"]["lr"],
            "sparsity_weight": config["model"]["sparsity_weight"],
            "tau_init": config["model"]["tau_init"],
            "host": config["servers"]["learner"]["host"],
            "port": config["servers"]["learner"]["port"],
            "device": config["servers"]["learner"]["device"],
            "checkpoint_dir": config["servers"]["learner"]["checkpoint_dir"],
            "checkpoint_every": config["servers"]["learner"].get("checkpoint_every", 500),
            "use_wandb": config["servers"]["learner"].get("use_wandb", False),
            "log_level": config["monitoring"]["log_level"],
        }
    
    @staticmethod
    def get_coordinator_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract coordinator configuration."""
        return {
            "collector_url": f"http://{config['servers']['collector']['host']}:{config['servers']['collector']['port']}",
            "learner_url": f"http://{config['servers']['learner']['host']}:{config['servers']['learner']['port']}",
            "queue_size": config["training"]["queue_size"],
            "replay_buffer_size": config["training"]["replay_buffer_size"],
            "replay_ratio": config["training"]["replay_ratio"],
            "enable_monitoring": config["monitoring"]["enable_monitoring"],
            "enable_wandb": config["monitoring"]["enable_wandb"],
            "plot_update_interval": config["monitoring"]["plot_update_interval"],
            "timeouts": config["timeouts"],
            "max_steps": config["training"]["max_steps"],
            "log_freq": config["training"]["log_freq"],
            "checkpoint_freq": config["training"]["checkpoint_freq"],
        }
    
    @staticmethod
    def print_config_summary(config: Dict[str, Any]) -> None:
        """Print a summary of the loaded configuration."""
        print("=" * 60)
        print(f"EXPERIMENT: {config.get('experiment_name', 'Unnamed')}")
        if config.get('description'):
            print(f"Description: {config['description']}")
        print("=" * 60)
        
        print(f"Model: d={config['model']['d']} → k={config['model']['k']}")
        print(f"Learning rate: {config['model']['lr']}")
        print(f"Sparsity weight: {config['model']['sparsity_weight']}")
        print(f"Initial tau: {config['model']['tau_init']}")
        print()
        
        print(f"Dataset: {config['data']['dataset_path']}")
        print(f"Attribute prompts: {config['data']['attribute_prompts_path']}")
        print(f"VLLM model: {config['vllm']['model_name']}")
        print(f"VLLM server: {config['vllm']['server_url']}")
        print()
        
        print(f"Collector: {config['servers']['collector']['host']}:{config['servers']['collector']['port']} ({config['servers']['collector']['device']})")
        print(f"Learner: {config['servers']['learner']['host']}:{config['servers']['learner']['port']} ({config['servers']['learner']['device']})")
        print()
        
        print(f"Training: {config['training']['max_steps']} steps")
        print(f"Batch size: {config['training']['users_per_batch']} users × {config['training']['samples_per_user']} samples")
        print(f"Replay buffer: {config['training']['replay_buffer_size']} (ratio: {config['training']['replay_ratio']})")
        print(f"W&B: {config['monitoring']['enable_wandb']}")
        print("=" * 60)


def load_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to load and process config."""
    loader = ConfigLoader()
    config = loader.load_config(config_path)
    config = loader.resolve_auto_d(config)
    config = loader.apply_environment_overrides(config)
    return config


def main():
    """CLI for testing config loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test config loading")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--component", choices=["collector", "learner", "coordinator"], 
                       help="Extract specific component config")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        
        if args.component == "collector":
            component_config = ConfigLoader.get_collector_config(config)
            print(json.dumps(component_config, indent=2))
        elif args.component == "learner":
            component_config = ConfigLoader.get_learner_config(config)
            print(json.dumps(component_config, indent=2))
        elif args.component == "coordinator":
            component_config = ConfigLoader.get_coordinator_config(config)
            print(json.dumps(component_config, indent=2))
        else:
            ConfigLoader.print_config_summary(config)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()