#!/usr/bin/env python3
"""
Test coordinator with both collector and learner servers to verify the full distributed system.
"""

import subprocess
import time
import logging
import requests
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoordinatorTester:
    def __init__(self, collector_url="http://localhost:8001", learner_url="http://localhost:8002"):
        self.collector_url = collector_url.rstrip('/')
        self.learner_url = learner_url.rstrip('/')
        self.processes = []
    
    def start_collector_server(self, dataset_path, attribute_prompts_path, d=400, port=8001):
        """Start collector server"""
        cmd = [
            sys.executable, "collector_server.py",
            "--d", str(d),
            "--dataset-path", dataset_path,
            "--attribute-prompts-path", attribute_prompts_path,
            "--vllm-model", "meta-llama/Llama-3.2-1B-Instruct",
            "--gpu-memory-util", "0.4",
            "--host", "localhost",
            "--port", str(port),
            "--device", "cuda:0",
            "--log-level", "INFO"
        ]
        
        logging.info(f"Starting collector server: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        self.processes.append(("collector", process))
        return process
    
    def start_learner_server(self, d=400, k=50, port=8002):
        """Start learner server"""
        cmd = [
            sys.executable, "learner_server.py",
            "--d", str(d),
            "--k", str(k),
            "--lr", "0.001",
            "--sparsity-weight", "0.0",
            "--tau-init", "1.0",
            "--host", "localhost",
            "--port", str(port),
            "--device", "cuda:0",  # Same GPU for testing
            "--log-level", "INFO"
        ]
        
        logging.info(f"Starting learner server: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        self.processes.append(("learner", process))
        return process
    
    def start_coordinator(self, steps=50):
        """Start coordinator"""
        cmd = [
            sys.executable, "coordinator.py",
            "--collector-url", self.collector_url,
            "--learner-url", self.learner_url,
            "--users-per-batch", "2",
            "--samples-per-user", "4", 
            "--steps", str(steps),
            "--replay-buffer-size", "100"
        ]
        
        logging.info(f"Starting coordinator: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        self.processes.append(("coordinator", process))
        return process
    
    def wait_for_server(self, url, name, timeout=60):
        """Wait for server to be ready"""
        logging.info(f"Waiting for {name} server at {url}...")
        
        for attempt in range(0, timeout, 5):
            time.sleep(5)
            try:
                response = requests.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    logging.info(f"✅ {name} server ready after ~{attempt + 5} seconds!")
                    return True
            except:
                pass
            
            if attempt % 15 == 0:
                logging.info(f"Still waiting for {name} server... ({attempt + 5}/{timeout}s)")
        
        logging.error(f"❌ {name} server failed to start within {timeout} seconds")
        return False
    
    def test_individual_servers(self):
        """Test that individual servers are working"""
        logging.info("=== Testing Individual Servers ===")
        
        # Test collector
        try:
            response = requests.get(f"{self.collector_url}/status", timeout=10.0)
            if response.status_code == 200:
                status = response.json()
                logging.info(f"✅ Collector status: {status}")
            else:
                logging.error(f"❌ Collector status failed: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"❌ Collector error: {e}")
            return False
        
        # Test learner
        try:
            response = requests.get(f"{self.learner_url}/status", timeout=10.0)
            if response.status_code == 200:
                status = response.json()
                logging.info(f"✅ Learner status: {status}")
            else:
                logging.error(f"❌ Learner status failed: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"❌ Learner error: {e}")
            return False
        
        return True
    
    def test_collector_learner_communication(self):
        """Test communication between collector and learner"""
        logging.info("=== Testing Collector-Learner Communication ===")
        
        # Get initial parameters from learner
        try:
            response = requests.get(f"{self.learner_url}/get_params", timeout=10.0)
            if response.status_code != 200:
                logging.error("❌ Failed to get initial parameters")
                return False
            
            params = response.json()
            if not params["success"]:
                logging.error(f"❌ Parameters request failed: {params.get('error')}")
                return False
            
            logging.info(f"✅ Got initial parameters: step={params['step']}, tau={params['tau']:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Parameter request error: {e}")
            return False
        
        # Test collector batch generation
        try:
            request_data = {
                "users_per_batch": 1,
                "samples_per_user": 2,
                "behavior_logits": params['mask_logits'],  # Use all mask_logits
                "tau": params['tau']
            }
            
            logging.info("Testing collector batch generation...")
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                json=request_data,
                timeout=120.0  # VLLM can be slow
            )
            
            if response.status_code != 200:
                logging.error(f"❌ Batch generation failed: {response.status_code}")
                return False
            
            result = response.json()
            if not result["success"]:
                logging.error(f"❌ Batch generation failed: {result.get('error')}")
                return False
            
            logging.info(f"✅ Collector generated batch: mask_sparsity={sum(result['m_hard'])}, reward_shape={len(result['R'])}x{len(result['R'][0])}")
            
            # Test learner training step
            train_request = {
                "m_hard": result['m_hard'],
                "R": result['R'],
                "user_data": result['user_data'],
                "success": True
            }
            
            response = requests.post(
                f"{self.learner_url}/train_step",
                json=train_request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logging.error(f"❌ Training step failed: {response.status_code}")
                return False
            
            train_result = response.json()
            if not train_result["success"]:
                logging.error(f"❌ Training step failed: {train_result.get('error')}")
                return False
            
            logging.info(f"✅ Learner training step: loss={train_result['loss']:.4f}, active_attrs={train_result['active_attributes']:.1f}")
            
        except Exception as e:
            logging.error(f"❌ Communication test error: {e}")
            return False
        
        return True
    
    def monitor_training_progress(self, coordinator_process, monitor_time=300):
        """Monitor training progress for a specified time"""
        logging.info(f"=== Monitoring Training Progress for {monitor_time}s ===")
        
        start_time = time.time()
        metrics = {
            'timestamps': [],
            'learner_steps': [],
            'active_features': [],
            'losses': []
        }
        
        while time.time() - start_time < monitor_time:
            if coordinator_process.poll() is not None:
                logging.info("Coordinator process finished")
                break
            
            try:
                # Get learner status
                response = requests.get(f"{self.learner_url}/status", timeout=5.0)
                if response.status_code == 200:
                    status = response.json()
                    
                    metrics['timestamps'].append(time.time() - start_time)
                    metrics['learner_steps'].append(status['current_step'])
                    metrics['active_features'].append(status['active_features'])
                    
                    # Try to get parameters for loss info (optional)
                    try:
                        params_response = requests.get(f"{self.learner_url}/get_params", timeout=5.0)
                        if params_response.status_code == 200:
                            params = params_response.json()
                            # We don't have loss in params, so use step as proxy
                            metrics['losses'].append(status['current_step'])
                    except:
                        metrics['losses'].append(0)
                    
                    if len(metrics['timestamps']) % 10 == 1:  # Log every 10th measurement
                        logging.info(f"Progress: Step {status['current_step']}, "
                                   f"Active features: {status['active_features']:.1f}")
            
            except Exception as e:
                logging.warning(f"Failed to get status: {e}")
            
            time.sleep(10)  # Check every 10 seconds
        
        # Create progress plot
        if metrics['timestamps']:
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                ax1.plot(metrics['timestamps'], metrics['learner_steps'])
                ax1.set_title('Training Steps Over Time')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Step')
                ax1.grid(True)
                
                ax2.plot(metrics['timestamps'], metrics['active_features'])
                ax2.set_title('Active Features Over Time')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Active Features')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig("coordinator_training_progress.png", dpi=150, bbox_inches='tight')
                logging.info("Saved training progress plot to coordinator_training_progress.png")
                plt.close()
                
            except Exception as e:
                logging.error(f"Failed to create progress plot: {e}")
        
        return len(metrics['timestamps']) > 0
    
    def cleanup(self):
        """Clean up all processes"""
        logging.info("Cleaning up processes...")
        
        for name, process in self.processes:
            if process.poll() is None:  # Still running
                logging.info(f"Terminating {name} process...")
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                    logging.info(f"{name} terminated gracefully")
                except subprocess.TimeoutExpired:
                    logging.warning(f"Force killing {name} process")
                    process.kill()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        logging.error(f"Failed to kill {name} process!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Full Coordinator System")
    parser.add_argument("--dataset-path", type=str, default="test_data.pkl", help="Dataset path")
    parser.add_argument("--attribute-prompts-path", type=str, default="test_attribute_prompts.json", 
                       help="Attribute prompts path")
    parser.add_argument("--d", type=int, default=3, help="Number of attributes (small for testing)")
    parser.add_argument("--steps", type=int, default=20, help="Number of coordinator steps")
    parser.add_argument("--monitor-time", type=int, default=120, help="Time to monitor training")
    parser.add_argument("--skip-coordinator", action="store_true", help="Skip coordinator test, just test servers")
    
    args = parser.parse_args()
    
    tester = CoordinatorTester()
    
    try:
        logging.info("=== Starting Full System Test ===")
        
        # Start servers
        tester.start_collector_server(args.dataset_path, args.attribute_prompts_path, args.d)
        tester.start_learner_server(args.d, k=10)  # Smaller k for testing
        
        # Wait for servers to be ready
        if not tester.wait_for_server(tester.collector_url, "collector"):
            return 1
        if not tester.wait_for_server(tester.learner_url, "learner"):
            return 1
        
        # Test individual servers
        if not tester.test_individual_servers():
            return 1
        
        # Test communication
        if not tester.test_collector_learner_communication():
            return 1
        
        if not args.skip_coordinator:
            # Start coordinator
            coordinator_process = tester.start_coordinator(args.steps)
            
            # Monitor training
            success = tester.monitor_training_progress(coordinator_process, args.monitor_time)
            
            if success:
                logging.info("🎉 Full system test successful!")
            else:
                logging.error("💥 Full system test failed!")
            
            return 0 if success else 1
        else:
            logging.info("🎉 Server communication test successful!")
            return 0
            
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Test error: {e}")
        return 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())