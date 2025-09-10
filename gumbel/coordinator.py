#!/usr/bin/env python3
"""
Coordinator: Start and manage both collector and learner servers.
Handles server lifecycle and communication between them.
"""

import asyncio
import logging
import argparse
import time
import subprocess
import sys
import requests
from pathlib import Path
from typing import Dict, Any, Optional

class ServerCoordinator:
    """Coordinates collector and learner servers"""
    
    def __init__(self, 
                 collector_args: Dict[str, Any],
                 learner_args: Dict[str, Any],
                 startup_wait: float = 10.0):
        
        self.collector_args = collector_args
        self.learner_args = learner_args
        self.startup_wait = startup_wait
        
        self.collector_process = None
        self.learner_process = None
        
        # Server URLs
        self.collector_url = f"http://localhost:{collector_args['port']}"
        self.learner_url = f"http://localhost:{learner_args['port']}"
        
        logging.info("ServerCoordinator initialized")
        logging.info(f"Collector: {self.collector_url}")
        logging.info(f"Learner: {self.learner_url}")
    
    def start_collector_server(self):
        """Start collector server subprocess"""
        cmd = [
            sys.executable, "collector_server.py",
            "--d", str(self.collector_args['d']),
            "--dataset-path", self.collector_args['dataset_path'],
            "--vllm-model", self.collector_args['vllm_model'],
            "--gpu-memory-util", str(self.collector_args['gpu_memory_util']),
            "--host", self.collector_args['host'],
            "--port", str(self.collector_args['port']),
            "--device", self.collector_args['device'],
            "--log-level", self.collector_args['log_level']
        ]
        
        logging.info(f"Starting collector server: {' '.join(cmd)}")
        self.collector_process = subprocess.Popen(cmd)
        return self.collector_process
    
    def start_learner_server(self):
        """Start learner server subprocess"""
        cmd = [
            sys.executable, "learner_server.py",
            "--d", str(self.learner_args['d']),
            "--k", str(self.learner_args['k']),
            "--lr", str(self.learner_args['lr']),
            "--sparsity-weight", str(self.learner_args['sparsity_weight']),
            "--tau-init", str(self.learner_args['tau_init']),
            "--collector-url", self.collector_url,
            "--host", self.learner_args['host'],
            "--port", str(self.learner_args['port']),
            "--device", self.learner_args['device'],
            "--checkpoint-dir", self.learner_args['checkpoint_dir'],
            "--log-level", self.learner_args['log_level']
        ]
        
        if self.learner_args.get('use_wandb', False):
            cmd.append("--use-wandb")
        
        logging.info(f"Starting learner server: {' '.join(cmd)}")
        self.learner_process = subprocess.Popen(cmd)
        return self.learner_process
    
    async def wait_for_server(self, url: str, name: str, max_wait: float = 30.0) -> bool:
        """Wait for server to be ready"""
        logging.info(f"Waiting for {name} server at {url}...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{url}/health", timeout=2.0)
                if response.status_code == 200:
                    health_data = response.json()
                    logging.info(f"{name} server ready: {health_data}")
                    return True
            except Exception as e:
                logging.debug(f"{name} server not ready yet: {e}")
            
            await asyncio.sleep(1.0)
        
        logging.error(f"{name} server failed to start within {max_wait}s")
        return False
    
    async def start_servers(self):
        """Start both servers and wait for them to be ready"""
        
        # Start collector first (has VLLM initialization)
        logging.info("=== Starting Collector Server ===")
        self.start_collector_server()
        
        # Wait for collector to be ready
        if not await self.wait_for_server(self.collector_url, "Collector", max_wait=60.0):
            raise RuntimeError("Collector server failed to start")
        
        # Start learner
        logging.info("=== Starting Learner Server ===") 
        self.start_learner_server()
        
        # Wait for learner to be ready
        if not await self.wait_for_server(self.learner_url, "Learner", max_wait=30.0):
            raise RuntimeError("Learner server failed to start")
        
        logging.info("=== Both servers ready ===")
    
    async def start_training(self, max_steps: int = 1000, log_freq: int = 20, checkpoint_freq: int = 100):
        """Start the training process on learner server"""
        logging.info(f"Starting distributed training for {max_steps} steps")
        
        try:
            response = requests.post(
                f"{self.learner_url}/start_training",
                json={
                    "max_steps": max_steps,
                    "log_freq": log_freq,
                    "checkpoint_freq": checkpoint_freq
                },
                timeout=max_steps * 2  # Generous timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    logging.info("Training completed successfully!")
                    logging.info(f"Final step: {result['final_step']}")
                    logging.info(f"Final active features: {result['final_active_features']:.1f}")
                    logging.info(f"Final sparsity ratio: {result['final_sparsity_ratio']:.3f}")
                    return True
                else:
                    logging.error(f"Training failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                logging.error(f"Failed to start training: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Error starting training: {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of both servers"""
        status = {"collector": {}, "learner": {}}
        
        try:
            resp = requests.get(f"{self.collector_url}/status", timeout=5.0)
            if resp.status_code == 200:
                status["collector"] = resp.json()
        except Exception as e:
            status["collector"] = {"error": str(e)}
        
        try:
            resp = requests.get(f"{self.learner_url}/status", timeout=5.0)
            if resp.status_code == 200:
                status["learner"] = resp.json()
        except Exception as e:
            status["learner"] = {"error": str(e)}
        
        return status
    
    def stop_servers(self):
        """Stop both server processes"""
        logging.info("Stopping servers...")
        
        if self.learner_process:
            logging.info("Stopping learner server...")
            self.learner_process.terminate()
            try:
                self.learner_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logging.warning("Force killing learner server")
                self.learner_process.kill()
        
        if self.collector_process:
            logging.info("Stopping collector server...")
            self.collector_process.terminate()
            try:
                self.collector_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logging.warning("Force killing collector server")
                self.collector_process.kill()
        
        logging.info("Servers stopped")
    
    async def run_training(self, max_steps: int = 1000, log_freq: int = 20, checkpoint_freq: int = 100):
        """Complete training workflow: start servers, run training, cleanup"""
        try:
            # Start servers
            await self.start_servers()
            
            # Start training
            success = await self.start_training(max_steps, log_freq, checkpoint_freq)
            
            if success:
                logging.info("=== Training Completed Successfully ===")
            else:
                logging.error("=== Training Failed ===")
            
            return success
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            return False
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            return False
        finally:
            self.stop_servers()

def main():
    parser = argparse.ArgumentParser(description="Distributed Sparse Attribute Learning Coordinator")
    
    # Model parameters
    parser.add_argument("--d", type=int, default=100, help="Number of attributes")
    parser.add_argument("--k", type=int, default=10, help="Number of components")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.1, help="Sparsity weight")
    parser.add_argument("--tau-init", type=float, default=1.0, help="Initial temperature")
    
    # Training parameters
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--log-freq", type=int, default=20, help="Logging frequency")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Checkpoint frequency")
    
    # Data parameters
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    
    # VLLM parameters
    parser.add_argument("--vllm-model", type=str, default="microsoft/DialoGPT-medium", help="VLLM model")
    parser.add_argument("--gpu-memory-util", type=float, default=0.8, help="GPU memory utilization")
    
    # System parameters
    parser.add_argument("--collector-device", type=str, default="cuda:0", help="Collector device")
    parser.add_argument("--learner-device", type=str, default="cuda:1", help="Learner device")
    parser.add_argument("--collector-port", type=int, default=8001, help="Collector port")
    parser.add_argument("--learner-port", type=int, default=8002, help="Learner port")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Prepare server configurations
    collector_args = {
        'd': args.d,
        'dataset_path': args.dataset_path,
        'vllm_model': args.vllm_model,
        'gpu_memory_util': args.gpu_memory_util,
        'host': '0.0.0.0',
        'port': args.collector_port,
        'device': args.collector_device,
        'log_level': args.log_level
    }
    
    learner_args = {
        'd': args.d,
        'k': args.k,
        'lr': args.lr,
        'sparsity_weight': args.sparsity_weight,
        'tau_init': args.tau_init,
        'host': '0.0.0.0',
        'port': args.learner_port,
        'device': args.learner_device,
        'checkpoint_dir': args.checkpoint_dir,
        'use_wandb': args.use_wandb,
        'log_level': args.log_level
    }
    
    # Create coordinator
    coordinator = ServerCoordinator(collector_args, learner_args)
    
    logging.info("=== Distributed Sparse Attribute Learning ===")
    logging.info(f"Dataset: {args.dataset_path}")
    logging.info(f"Model: {args.d} attributes -> {args.k} components")
    logging.info(f"Devices: Collector={args.collector_device}, Learner={args.learner_device}")
    logging.info(f"Training: {args.max_steps} steps")
    
    # Run training
    async def run():
        success = await coordinator.run_training(
            max_steps=args.max_steps,
            log_freq=args.log_freq,
            checkpoint_freq=args.checkpoint_freq
        )
        return success
    
    try:
        success = asyncio.run(run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logging.info("Coordinator interrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()