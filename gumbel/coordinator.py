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
from typing import Dict, Any, Optional, Tuple
import queue
import threading
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import json

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ServerCoordinator:
    """Coordinates collector and learner servers"""
    
    def __init__(self, 
                 collector_args: Dict[str, Any],
                 learner_args: Dict[str, Any],
                 queue_size: int = 100,
                 replay_buffer_size: int = 10000,
                 replay_ratio: float = 0.3,
                 startup_wait: float = 10.0,
                 enable_monitoring: bool = True,
                 enable_wandb: bool = False,
                 plot_update_interval: float = 10.0):
        
        self.collector_args = collector_args
        self.learner_args = learner_args
        self.startup_wait = startup_wait
        
        self.collector_process = None
        self.learner_process = None
        
        # Server URLs
        self.collector_url = f"http://{collector_args['host']}:{collector_args['port']}"
        self.learner_url = f"http://{learner_args['host']}:{learner_args['port']}"
        
        # Batch queue for producer-consumer pattern
        self.batch_queue = asyncio.Queue(maxsize=queue_size)
        self.training_active = False
        self.producer_task = None
        self.consumer_task = None
        
        # Replay buffer for improved sample efficiency
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_ratio = replay_ratio
        
        # Monitoring configuration
        self.enable_monitoring = enable_monitoring
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.plot_update_interval = plot_update_interval
        self.start_time = None
        
        # Metrics tracking
        self.metrics = {
            'timestamps': [],
            'steps': [],
            'losses': [],
            'reward_signals': [],
            'active_attributes': [],
            'temperatures': [],
            'queue_sizes': [],
            'replay_buffer_sizes': []
        }
        
        # Plotting
        self.fig = None
        self.axes = None
        self.monitoring_thread = None
        self.wandb_run = None
        
        # Training configuration
        self.users_per_batch = 4
        self.samples_per_user = 8
        
        logging.info("ServerCoordinator initialized with queue-based architecture and replay buffer")
        logging.info(f"Collector: {self.collector_url}")
        logging.info(f"Learner: {self.learner_url}")
        logging.info(f"Queue size: {queue_size}")
        logging.info(f"Replay buffer size: {replay_buffer_size}, replay ratio: {replay_ratio}")
    
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
        
        # Add attribute prompts path if provided
        if self.collector_args.get('attribute_prompts_path'):
            cmd.extend(["--attribute-prompts-path", self.collector_args['attribute_prompts_path']])
        
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
    
    def add_to_replay_buffer(self, m_hard: list, R: list, user_data: Dict[str, Any]):
        """Add batch data to replay buffer"""
        batch_size = len(R)
        
        # Store each sample individually
        for i in range(batch_size):
            sample = {
                'm_hard': m_hard,  # Same mask for all samples in batch
                'reward_vector': R[i],  # [d] reward vector for this sample
                'prompt': user_data['prompts'][i],
                'output': user_data['outputs'][i], 
                'user_id': user_data['user_ids'][i]
            }
            self.replay_buffer.append(sample)
        
        logging.debug(f"Added {batch_size} samples to replay buffer, total size: {len(self.replay_buffer)}")
    
    def sample_replay_data(self, target_batch_size: int) -> Optional[Dict[str, Any]]:
        """Sample replay data to mix with fresh data"""
        if len(self.replay_buffer) == 0:
            return None
        
        replay_size = min(int(target_batch_size * self.replay_ratio), len(self.replay_buffer))
        if replay_size == 0:
            return None
        
        # Sample random replay data
        replay_samples = random.sample(list(self.replay_buffer), replay_size)
        
        # Reconstruct batch format
        m_hard = replay_samples[0]['m_hard']  # Same mask for all
        R = [sample['reward_vector'] for sample in replay_samples]  # [replay_size, d]
        
        user_data = {
            'prompts': [sample['prompt'] for sample in replay_samples],
            'outputs': [sample['output'] for sample in replay_samples],
            'user_ids': [sample['user_id'] for sample in replay_samples]
        }
        
        replay_data = {
            'm_hard': m_hard,
            'R': R,
            'user_data': user_data,
            'success': True
        }
        
        logging.debug(f"Sampled {replay_size} replay samples from buffer")
        return replay_data
    
    def mix_fresh_and_replay(self, fresh_data: Dict[str, Any], replay_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Mix fresh data with replay data"""
        if replay_data is None:
            return fresh_data
        
        # Combine reward matrices
        combined_R = fresh_data['R'] + replay_data['R']
        
        # Combine user data
        combined_user_data = {
            'prompts': fresh_data['user_data']['prompts'] + replay_data['user_data']['prompts'],
            'outputs': fresh_data['user_data']['outputs'] + replay_data['user_data']['outputs'],
            'user_ids': fresh_data['user_data']['user_ids'] + replay_data['user_data']['user_ids']
        }
        
        mixed_data = {
            'm_hard': fresh_data['m_hard'],  # Use fresh mask
            'R': combined_R,
            'user_data': combined_user_data,
            'success': True
        }
        
        logging.debug(f"Mixed fresh batch (size={len(fresh_data['R'])}) with replay batch (size={len(replay_data['R'])})")
        return mixed_data
    
    def get_replay_stats(self) -> Dict[str, Any]:
        """Get replay buffer statistics"""
        if len(self.replay_buffer) == 0:
            return {
                'size': 0,
                'max_size': self.replay_buffer.maxlen,
                'utilization': 0.0,
                'replay_ratio': self.replay_ratio,
                'unique_users': 0
            }
        
        # Count unique users
        unique_users = len(set(sample['user_id'] for sample in self.replay_buffer))
        
        return {
            'size': len(self.replay_buffer),
            'max_size': self.replay_buffer.maxlen,
            'utilization': len(self.replay_buffer) / self.replay_buffer.maxlen,
            'replay_ratio': self.replay_ratio,
            'unique_users': unique_users
        }
    
    async def get_learner_params(self) -> Dict[str, Any]:
        """Get current model parameters from learner"""
        try:
            response = requests.get(f"{self.learner_url}/get_params", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Failed to get learner params: HTTP {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error getting learner params: {e}")
            return None
    
    async def call_collector_generate_batch(self, behavior_logits: list, tau: float) -> Dict[str, Any]:
        """Request batch generation from collector"""
        try:
            request_data = {
                "users_per_batch": self.users_per_batch,
                "samples_per_user": self.samples_per_user,
                "behavior_logits": behavior_logits,
                "tau": tau
            }
            
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Collector batch generation failed: HTTP {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error calling collector: {e}")
            return None
    
    async def call_learner_train_step(self, batch_data: Dict[str, Any]) -> bool:
        """Send batch to learner for training step"""
        try:
            response = requests.post(
                f"{self.learner_url}/train_step",
                json=batch_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                logging.error(f"Learner training step failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error calling learner: {e}")
            return False
    
    async def producer_loop(self):
        """Producer loop: continuously generate batches, mix with replay data, and add to queue"""
        logging.info("Starting producer loop with replay buffer...")
        
        while self.training_active:
            try:
                # Get current parameters from learner
                params_data = await self.get_learner_params()
                if params_data is None or not params_data.get("success", False):
                    logging.warning("Failed to get learner params, retrying...")
                    await asyncio.sleep(1.0)
                    continue
                
                behavior_logits = params_data["mask_logits"]
                tau = params_data["tau"]
                
                # Generate fresh batch from collector
                fresh_batch_data = await self.call_collector_generate_batch(behavior_logits, tau)
                if fresh_batch_data is None or not fresh_batch_data.get("success", False):
                    logging.warning("Failed to generate batch, retrying...")
                    await asyncio.sleep(1.0)
                    continue
                
                # Add fresh data to replay buffer for future use
                self.add_to_replay_buffer(
                    fresh_batch_data['m_hard'], 
                    fresh_batch_data['R'], 
                    fresh_batch_data['user_data']
                )
                
                # Sample replay data to mix with fresh data
                replay_data = self.sample_replay_data(len(fresh_batch_data['R']))
                
                # Mix fresh and replay data
                mixed_batch_data = self.mix_fresh_and_replay(fresh_batch_data, replay_data)
                
                # Add mixed batch to queue (this will block if queue is full)
                await self.batch_queue.put(mixed_batch_data)
                
                replay_stats = self.get_replay_stats()
                logging.debug(f"Added mixed batch to queue (fresh+replay), queue size: {self.batch_queue.qsize()}, "
                             f"replay buffer: {replay_stats['size']}/{replay_stats['max_size']}")
                
            except Exception as e:
                logging.error(f"Error in producer loop: {e}")
                await asyncio.sleep(1.0)
        
        logging.info("Producer loop stopped")
    
    async def consumer_loop(self):
        """Consumer loop: continuously process batches from queue"""
        logging.info("Starting consumer loop...")
        step = 0
        
        while self.training_active:
            try:
                # Get batch from queue (this will block until available)
                batch_data = await self.batch_queue.get()
                
                # Send to learner for training
                success = await self.call_learner_train_step(batch_data)
                if success:
                    step += 1
                    if step % 10 == 0:
                        logging.info(f"Completed training step {step}, queue size: {self.batch_queue.qsize()}")
                else:
                    logging.warning(f"Training step {step} failed")
                
                # Mark task as done
                self.batch_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1.0)
        
        logging.info(f"Consumer loop stopped after {step} steps")
    
    async def update_metrics(self, step: int, loss: float = None, reward_signal: float = None, 
                           active_attributes: float = None, temperature: float = None):
        """Update metrics tracking"""
        if not self.enable_monitoring:
            return
            
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        timestamp = current_time - self.start_time
        
        self.metrics['timestamps'].append(timestamp)
        self.metrics['steps'].append(step)
        self.metrics['losses'].append(loss or 0.0)
        self.metrics['reward_signals'].append(reward_signal or 0.0)
        self.metrics['active_attributes'].append(active_attributes or 0.0)
        self.metrics['temperatures'].append(temperature or 1.0)
        self.metrics['queue_sizes'].append(self.batch_queue.qsize())
        self.metrics['replay_buffer_sizes'].append(len(self.replay_buffer))
        
        # Log to wandb if enabled
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.log({
                'step': step,
                'loss': loss or 0.0,
                'reward_signal': reward_signal or 0.0,
                'active_attributes': active_attributes or 0.0,
                'temperature': temperature or 1.0,
                'queue_size': self.batch_queue.qsize(),
                'replay_buffer_size': len(self.replay_buffer),
                'timestamp': timestamp
            }, step=step)
    
    def setup_monitoring(self):
        """Setup monitoring components"""
        if not self.enable_monitoring:
            return
            
        # Initialize wandb if enabled
        if self.enable_wandb:
            try:
                self.wandb_run = wandb.init(
                    project="distributed-sparse-attributes",
                    name=f"coordinator-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={
                        'users_per_batch': self.users_per_batch,
                        'samples_per_user': self.samples_per_user,
                        'replay_buffer_size': self.replay_buffer.maxlen,
                        'replay_ratio': self.replay_ratio,
                        'collector_args': self.collector_args,
                        'learner_args': self.learner_args
                    }
                )
                logging.info("Wandb monitoring initialized")
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {e}")
                self.enable_wandb = False
        
        # Setup matplotlib for live plotting
        if self.enable_monitoring:
            try:
                plt.ion()  # Interactive mode
                self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
                self.fig.suptitle('Real-time Training Monitoring', fontsize=14)
                
                # Configure subplots
                self.axes[0, 0].set_title('Training Loss')
                self.axes[0, 0].set_xlabel('Time (s)')
                self.axes[0, 0].set_ylabel('Loss')
                self.axes[0, 0].grid(True)
                
                self.axes[0, 1].set_title('Active Attributes')
                self.axes[0, 1].set_xlabel('Time (s)')
                self.axes[0, 1].set_ylabel('Count')
                self.axes[0, 1].grid(True)
                
                self.axes[1, 0].set_title('Queue & Buffer Status')
                self.axes[1, 0].set_xlabel('Time (s)')
                self.axes[1, 0].set_ylabel('Size')
                self.axes[1, 0].grid(True)
                
                self.axes[1, 1].set_title('Temperature')
                self.axes[1, 1].set_xlabel('Time (s)')
                self.axes[1, 1].set_ylabel('Temperature')
                self.axes[1, 1].grid(True)
                
                plt.tight_layout()
                logging.info("Live plotting initialized")
            except Exception as e:
                logging.error(f"Failed to setup live plotting: {e}")
    
    def update_plots(self):
        """Update live plots"""
        if not self.enable_monitoring or self.fig is None or not self.metrics['timestamps']:
            return
        
        try:
            timestamps = self.metrics['timestamps']
            
            # Clear and update each subplot
            self.axes[0, 0].clear()
            self.axes[0, 0].plot(timestamps, self.metrics['losses'], 'b-', alpha=0.7)
            self.axes[0, 0].set_title('Training Loss')
            self.axes[0, 0].set_xlabel('Time (s)')
            self.axes[0, 0].set_ylabel('Loss')
            self.axes[0, 0].grid(True)
            
            self.axes[0, 1].clear()
            self.axes[0, 1].plot(timestamps, self.metrics['active_attributes'], 'g-', alpha=0.7)
            self.axes[0, 1].set_title('Active Attributes')
            self.axes[0, 1].set_xlabel('Time (s)')
            self.axes[0, 1].set_ylabel('Count')
            self.axes[0, 1].grid(True)
            
            self.axes[1, 0].clear()
            self.axes[1, 0].plot(timestamps, self.metrics['queue_sizes'], 'r-', alpha=0.7, label='Queue')
            self.axes[1, 0].plot(timestamps, self.metrics['replay_buffer_sizes'], 'orange', alpha=0.7, label='Replay Buffer')
            self.axes[1, 0].set_title('Queue & Buffer Status')
            self.axes[1, 0].set_xlabel('Time (s)')
            self.axes[1, 0].set_ylabel('Size')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
            
            self.axes[1, 1].clear()
            self.axes[1, 1].plot(timestamps, self.metrics['temperatures'], 'purple', alpha=0.7)
            self.axes[1, 1].set_title('Temperature')
            self.axes[1, 1].set_xlabel('Time (s)')
            self.axes[1, 1].set_ylabel('Temperature')
            self.axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # Small pause to update display
            
        except Exception as e:
            logging.debug(f"Failed to update plots: {e}")
    
    def start_monitoring_thread(self):
        """Start background monitoring thread"""
        if not self.enable_monitoring:
            return
            
        def monitoring_loop():
            last_plot_update = time.time()
            
            while self.training_active:
                try:
                    current_time = time.time()
                    
                    # Update plots at specified interval
                    if current_time - last_plot_update >= self.plot_update_interval:
                        self.update_plots()
                        last_plot_update = current_time
                    
                    time.sleep(1.0)  # Check every second
                    
                except Exception as e:
                    logging.debug(f"Error in monitoring thread: {e}")
                    time.sleep(5.0)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Monitoring thread started")
    
    def save_monitoring_results(self):
        """Save final monitoring results"""
        if not self.enable_monitoring or not self.metrics['timestamps']:
            return
            
        try:
            # Save metrics to JSON
            metrics_file = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logging.info(f"Saved training metrics to {metrics_file}")
            
            # Save final plot
            if self.fig is not None:
                plot_file = f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                logging.info(f"Saved training plot to {plot_file}")
                
                if self.enable_wandb and self.wandb_run:
                    self.wandb_run.log({"final_training_plot": wandb.Image(plot_file)})
            
        except Exception as e:
            logging.error(f"Failed to save monitoring results: {e}")
    
    async def start_async_training(self, max_steps: int = 1000, log_freq: int = 20, checkpoint_freq: int = 100):
        """Start async distributed training with producer-consumer pattern"""
        logging.info(f"Starting async distributed training for {max_steps} steps")
        
        try:
            # Setup monitoring
            self.setup_monitoring()
            self.start_monitoring_thread()
            
            # Start training mode
            self.training_active = True
            
            # Start producer and consumer tasks
            self.producer_task = asyncio.create_task(self.producer_loop())
            self.consumer_task = asyncio.create_task(self.consumer_loop())
            
            logging.info("Producer and consumer loops started")
            
            # Monitor progress and stop after max_steps
            step = 0
            last_metrics_update = 0
            
            while step < max_steps and self.training_active:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                # Get status from learner to check progress
                try:
                    response = requests.get(f"{self.learner_url}/status", timeout=5.0)
                    if response.status_code == 200:
                        status = response.json()
                        current_step = status.get("current_step", step)
                        
                        if current_step > step:
                            step = current_step
                            
                            # Get additional metrics for monitoring
                            try:
                                params_response = requests.get(f"{self.learner_url}/get_params", timeout=5.0)
                                if params_response.status_code == 200:
                                    params = params_response.json()
                                    temperature = params.get('tau', 1.0) if params.get('success') else 1.0
                                else:
                                    temperature = 1.0
                            except:
                                temperature = 1.0
                            
                            active_features = status.get("active_features", 0)
                            
                            # Update metrics (we don't have loss from status, so use placeholder)
                            await self.update_metrics(
                                step=step,
                                loss=None,  # Will be set to 0.0 in update_metrics
                                reward_signal=None,
                                active_attributes=active_features,
                                temperature=temperature
                            )
                            
                            if step % log_freq == 0:
                                replay_stats = self.get_replay_stats()
                                logging.info(f"Step {step}: active_features={active_features:.1f}, "
                                           f"temp={temperature:.3f}, "
                                           f"queue_size={self.batch_queue.qsize()}, "
                                           f"replay_buffer={replay_stats['size']}/{replay_stats['max_size']} "
                                           f"({100*replay_stats['utilization']:.1f}% full)")
                except Exception as e:
                    logging.debug(f"Could not get learner status: {e}")
            
            # Stop training
            self.training_active = False
            
            # Wait for tasks to complete
            if self.producer_task:
                await self.producer_task
            if self.consumer_task:
                await self.consumer_task
            
            # Save monitoring results
            self.save_monitoring_results()
            
            # Clean up wandb
            if self.enable_wandb and self.wandb_run:
                self.wandb_run.finish()
            
            logging.info(f"Training completed after {step} steps")
            return True
            
        except Exception as e:
            logging.error(f"Error in async training: {e}")
            self.training_active = False
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
        """Complete training workflow: start servers, run async training, cleanup"""
        try:
            # Start servers
            await self.start_servers()
            
            # Start async training with producer-consumer pattern
            success = await self.start_async_training(max_steps, log_freq, checkpoint_freq)
            
            if success:
                logging.info("=== Training Completed Successfully ===")
            else:
                logging.error("=== Training Failed ===")
            
            return success
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.training_active = False
            return False
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            self.training_active = False
            return False
        finally:
            # Ensure training is stopped
            self.training_active = False
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
    parser.add_argument("--queue-size", type=int, default=100, help="Batch queue size")
    
    # Replay buffer parameters
    parser.add_argument("--replay-buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--replay-ratio", type=float, default=0.3, help="Fraction of batch from replay buffer")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    # Monitoring parameters
    parser.add_argument("--enable-monitoring", action="store_true", default=True, help="Enable real-time monitoring")
    parser.add_argument("--enable-wandb-coordinator", action="store_true", help="Enable wandb for coordinator")
    parser.add_argument("--plot-update-interval", type=float, default=10.0, help="Plot update interval in seconds")
    parser.add_argument("--disable-monitoring", action="store_true", help="Disable monitoring completely")
    
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
    coordinator = ServerCoordinator(
        collector_args, learner_args, 
        queue_size=args.queue_size,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        enable_monitoring=not args.disable_monitoring,
        enable_wandb=args.enable_wandb_coordinator,
        plot_update_interval=args.plot_update_interval
    )
    
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