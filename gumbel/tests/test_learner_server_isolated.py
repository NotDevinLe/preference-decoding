#!/usr/bin/env python3
"""
Test isolated learner server with real reward data to verify HTTP API training works.
"""

import requests
import time
import logging
import numpy as np
import subprocess
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LearnerServerTester:
    def __init__(self, learner_url: str = "http://localhost:8002"):
        self.learner_url = learner_url.rstrip('/')
        
    def test_health(self):
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.learner_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    def get_params(self):
        """Get current model parameters"""
        try:
            response = requests.get(f"{self.learner_url}/get_params", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logging.error(f"Error getting params: {e}")
            return None
    
    def train_step(self, m_hard, R, user_data):
        """Send training step to learner"""
        try:
            request_data = {
                "m_hard": m_hard,
                "R": R,
                "user_data": user_data,
                "success": True
            }
            
            response = requests.post(
                f"{self.learner_url}/train_step",
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Training step failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error in training step: {e}")
            return None
    
    def run_training_simulation(self, reward_path="../data/reward_matrix_flexible.npz", num_steps=100, 
                               use_wandb=False, save_plots=True):
        """Simulate training with real reward data"""
        
        # Load real reward data
        logging.info(f"Loading reward data from {reward_path}")
        try:
            reward_data = np.load(reward_path)
            X = reward_data['Y_chosen']  # [num_samples, d]
            logging.info(f"Loaded Y_chosen shape: {X.shape}")
            num_samples, d = X.shape
        except Exception as e:
            logging.error(f"Failed to load reward data: {e}")
            return False
        
        # Get initial parameters
        initial_params = self.get_params()
        if not initial_params or not initial_params["success"]:
            logging.error("Failed to get initial parameters")
            return False
        
        logging.info(f"Initial step: {initial_params['step']}")
        logging.info(f"Initial temperature: {initial_params['tau']:.4f}")
        logging.info(f"Mask logits range: [{min(initial_params['mask_logits']):.6f}, {max(initial_params['mask_logits']):.6f}]")
        
        # Initialize wandb if requested
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="sparse-learner-server-test",
                config={
                    "num_steps": num_steps,
                    "d": d,
                    "batch_size": 32,
                    "reward_data": reward_path
                }
            )
        
        # Training tracking
        batch_size = 32
        successful_steps = 0
        
        # Lists to track metrics
        steps_list = []
        losses = []
        reward_signals = []
        active_attributes = []
        temperatures = []
        
        for step in range(num_steps):
            # Sample batch from real data
            sample_indices = np.random.choice(num_samples, size=batch_size, replace=True)
            R_batch = X[sample_indices].tolist()  # [batch_size, d]
            
            # Create a simple hard mask (random for now - normally this would come from collector)
            m_hard = np.random.binomial(1, 0.3, d).tolist()  # 30% sparsity
            
            # Create dummy user data
            user_data = {
                "prompts": [f"Sample prompt {i}" for i in range(batch_size)],
                "outputs": [f"Sample output {i}" for i in range(batch_size)], 
                "user_ids": [f"user_{i % 10}" for i in range(batch_size)]
            }
            
            # Send training step
            result = self.train_step(m_hard, R_batch, user_data)
            
            if result and result["success"]:
                successful_steps += 1
                
                # Track metrics
                steps_list.append(step)
                losses.append(result['loss'])
                reward_signals.append(result['reward_signal'])
                active_attributes.append(result['active_attributes'])
                
                # Get current temperature from params
                current_params = self.get_params()
                if current_params and current_params["success"]:
                    temperatures.append(current_params['tau'])
                else:
                    temperatures.append(temperatures[-1] if temperatures else 1.0)
                
                # Log to wandb if enabled
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "loss": result['loss'],
                        "reward_signal": result['reward_signal'], 
                        "active_attributes": result['active_attributes'],
                        "temperature": temperatures[-1],
                        "step": step
                    })
                
                if step % 20 == 0:
                    logging.info(f"Step {step}: Loss={result['loss']:.4f}, "
                               f"Reward={result['reward_signal']:.4f}, "
                               f"Active={result['active_attributes']:.1f}, "
                               f"Temp={temperatures[-1]:.3f}")
            else:
                logging.error(f"Step {step} failed")
                if result:
                    logging.error(f"Error: {result.get('error', 'Unknown')}")
        
        # Get final parameters
        final_params = self.get_params()
        if final_params and final_params["success"]:
            logging.info(f"\n=== TRAINING RESULTS ===")
            logging.info(f"Successful steps: {successful_steps}/{num_steps}")
            logging.info(f"Final step: {final_params['step']}")
            logging.info(f"Final temperature: {final_params['tau']:.4f}")
            
            # Check if parameters changed
            initial_logits = initial_params['mask_logits']
            final_logits = final_params['mask_logits']
            
            changes = [abs(f - i) for f, i in zip(final_logits, initial_logits)]
            max_change = max(changes)
            mean_change = sum(changes) / len(changes)
            
            logging.info(f"Parameter changes: max={max_change:.6f}, mean={mean_change:.6f}")
            
            if max_change > 1e-6:
                logging.info("✅ Parameters updated during training!")
                
                # Show top changed parameters
                top_changes = sorted(enumerate(changes), key=lambda x: x[1], reverse=True)[:10]
                logging.info("Top 10 parameter changes:")
                for idx, change in top_changes:
                    logging.info(f"  Param {idx}: {initial_logits[idx]:.6f} → {final_logits[idx]:.6f} (Δ={change:.6f})")
                
                # Create training plots
                if save_plots and losses:
                    self.create_training_plots(steps_list, losses, reward_signals, 
                                             active_attributes, temperatures, use_wandb)
                
                return True
            else:
                logging.warning("❌ Parameters did not change significantly")
                if save_plots and losses:
                    self.create_training_plots(steps_list, losses, reward_signals, 
                                             active_attributes, temperatures, use_wandb)
                return False
        
        return successful_steps > num_steps * 0.8  # 80% success rate
    
    def create_training_plots(self, steps, losses, reward_signals, active_attributes, temperatures, use_wandb=False):
        """Create and save training plots"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Loss curve
            ax1.plot(steps, losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # Reward signal
            ax2.plot(steps, reward_signals)
            ax2.set_title('Reward Signal')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.grid(True)
            
            # Active attributes
            ax3.plot(steps, active_attributes)
            ax3.set_title('Active Attributes')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Count')
            ax3.grid(True)
            
            # Temperature
            ax4.plot(steps, temperatures)
            ax4.set_title('Temperature')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Temperature')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = "learner_server_training.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved training plots to {plot_path}")
            
            # Log to wandb if enabled
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"training_curves": wandb.Image(plot_path)})
            
            plt.close()
            
        except Exception as e:
            logging.error(f"Failed to create plots: {e}")

def start_learner_server(d: int = 400, k: int = 50, port: int = 8002):
    """Start learner server for testing"""
    cmd = [
        sys.executable, "learner_server.py",
        "--d", str(d),
        "--k", str(k),
        "--lr", "0.001",
        "--sparsity-weight", "0.0",  # Match original gumbel.py default
        "--tau-init", "1.0",
        "--host", "localhost",
        "--port", str(port),
        "--device", "cuda:0",
        "--log-level", "INFO"
    ]
    
    logging.info(f"Starting learner server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Isolated Learner Server")
    parser.add_argument("--reward-path", type=str, default="../data/reward_matrix_flexible.npz", 
                       help="Path to reward data")
    parser.add_argument("--d", type=int, default=400, help="Number of attributes")
    parser.add_argument("--k", type=int, default=50, help="Number of components")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--port", type=int, default=8002, help="Learner port")
    parser.add_argument("--start-server", action="store_true", help="Start learner server automatically")
    parser.add_argument("--wait-time", type=int, default=30, help="Time to wait for server startup")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    
    args = parser.parse_args()
    
    learner_process = None
    
    try:
        if args.start_server:
            logging.info("Starting learner server...")
            learner_process = start_learner_server(args.d, args.k, args.port)
            
            # Wait for server to be ready
            tester = LearnerServerTester(f"http://localhost:{args.port}")
            logging.info(f"Waiting up to {args.wait_time} seconds for server to be ready...")
            
            ready = False
            for attempt in range(0, args.wait_time, 5):
                time.sleep(5)
                if tester.test_health():
                    logging.info(f"✅ Server ready after ~{attempt + 5} seconds!")
                    ready = True
                    break
                elif attempt % 15 == 0:
                    logging.info(f"Still waiting for server... ({attempt + 5}/{args.wait_time}s)")
            
            if not ready:
                logging.error(f"❌ Server failed to start within {args.wait_time} seconds")
                return 1
        else:
            tester = LearnerServerTester(f"http://localhost:{args.port}")

        # Run training simulation
        logging.info("=== Starting Training Simulation ===")
        success = tester.run_training_simulation(args.reward_path, args.steps, 
                                                use_wandb=args.wandb, save_plots=not args.no_plots)
        
        if success:
            logging.info("🎉 Training simulation successful!")
        else:
            logging.error("💥 Training simulation failed!")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Test error: {e}")
        return 1
    finally:
        if learner_process:
            logging.info("Stopping learner server...")
            learner_process.terminate()
            try:
                learner_process.wait(timeout=5.0)
                logging.info("Learner server stopped gracefully")
            except subprocess.TimeoutExpired:
                logging.warning("Force killing learner server")
                learner_process.kill()

if __name__ == "__main__":
    sys.exit(main())