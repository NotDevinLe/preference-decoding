#!/usr/bin/env python3
"""
Test script for learner server to verify training functionality.
Tests parameter retrieval, training steps, and loss computation.
"""

import requests
import time
import logging
import json
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LearnerTester:
    def __init__(self, learner_url: str = "http://localhost:8002"):
        self.learner_url = learner_url.rstrip('/')
        
    def test_health(self):
        """Test health endpoint"""
        logging.info("Testing learner health endpoint...")
        try:
            response = requests.get(f"{self.learner_url}/health", timeout=5.0)
            if response.status_code == 200:
                health_data = response.json()
                logging.info(f"✅ Health check passed: {health_data}")
                return True
            else:
                logging.error(f"❌ Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"❌ Health check error: {e}")
            return False
    
    def test_status(self):
        """Test status endpoint"""
        logging.info("Testing learner status endpoint...")
        try:
            response = requests.get(f"{self.learner_url}/status", timeout=5.0)
            if response.status_code == 200:
                status_data = response.json()
                logging.info(f"✅ Status check passed: {status_data}")
                return True
            else:
                logging.error(f"❌ Status check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"❌ Status check error: {e}")
            return False
    
    def test_get_params(self, d: int = 3):
        """Test parameter retrieval"""
        logging.info("Testing parameter retrieval...")
        try:
            response = requests.get(f"{self.learner_url}/get_params", timeout=10.0)
            if response.status_code == 200:
                params_data = response.json()
                if params_data["success"]:
                    logging.info("✅ Parameter retrieval successful!")
                    logging.info(f"  Current step: {params_data['step']}")
                    logging.info(f"  Temperature: {params_data['tau']:.4f}")
                    logging.info(f"  Mask logits: {[f'{x:.4f}' for x in params_data['mask_logits']]}")
                    
                    if len(params_data['mask_logits']) == d:
                        logging.info(f"✅ Correct mask dimension: {d}")
                    else:
                        logging.warning(f"⚠️ Mask dimension mismatch: expected {d}, got {len(params_data['mask_logits'])}")
                    
                    return True, params_data
                else:
                    logging.error(f"❌ Parameter retrieval failed: {params_data.get('error', 'Unknown error')}")
                    return False, None
            else:
                logging.error(f"❌ Parameter retrieval HTTP error: {response.status_code}")
                return False, None
        except Exception as e:
            logging.error(f"❌ Parameter retrieval error: {e}")
            return False, None
    
    def test_train_step(self, d: int = 3):
        """Test training step with sample data"""
        logging.info("Testing training step...")
        try:
            # Create sample training data (using data from collector test)
            train_request = {
                "m_hard": [1.0, 1.0, 0.0],  # Mask with 2 active attributes
                "R": [
                    [0.7085, -0.6202, -0.2121],  # Pirate response
                    [0.0995, 0.6319, 1.0810],    # Academic response  
                    [0.1348, 0.3329, 0.1254]     # Teen response
                ],
                "user_data": {
                    "prompts": ["Tell me about treasure hunting"] * 3,
                    "outputs": [
                        "Arrr, matey! Treasure hunting be the finest adventure on the seven seas!",
                        "Treasure hunting is an archaeological practice involving systematic search.",
                        "OMG treasure hunting is sooo cool! Like, you get to dig around!"
                    ],
                    "user_ids": ["user1", "user1", "user1"]
                },
                "success": True
            }
            
            logging.info(f"Sending training request with:")
            logging.info(f"  Mask: {train_request['m_hard']}")
            logging.info(f"  Reward matrix shape: {len(train_request['R'])}x{len(train_request['R'][0])}")
            logging.info(f"  Active attributes: {sum(train_request['m_hard'])}")
            
            response = requests.post(
                f"{self.learner_url}/train_step",
                json=train_request,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    logging.info("✅ Training step successful!")
                    logging.info(f"  Step: {result['step']}")
                    logging.info(f"  Loss: {result['loss']:.6f}")
                    logging.info(f"  Reward signal: {result['reward_signal']:.6f}")
                    logging.info(f"  Active attributes: {result['active_attributes']:.1f}")
                    
                    # Analyze results
                    logging.info("\n=== TRAINING ANALYSIS ===")
                    expected_active = sum(train_request['m_hard'])
                    if abs(result['active_attributes'] - expected_active) < 0.1:
                        logging.info(f"✅ Active attributes correct: {result['active_attributes']:.1f}")
                    else:
                        logging.warning(f"⚠️ Active attributes mismatch: expected {expected_active}, got {result['active_attributes']:.1f}")
                    
                    # Check if reward signal makes sense (should be positive for good rewards)
                    total_reward = sum(sum(row) for row in train_request['R'])
                    avg_reward = total_reward / (len(train_request['R']) * len(train_request['R'][0]))
                    logging.info(f"Average reward in input: {avg_reward:.4f}")
                    logging.info(f"Computed reward signal: {result['reward_signal']:.4f}")
                    
                    return True
                else:
                    logging.error(f"❌ Training step failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                logging.error(f"❌ Training step HTTP error: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Training step error: {e}")
            return False
    
    def test_multiple_steps(self, num_steps: int = 3):
        """Test multiple training steps to verify parameter updates"""
        logging.info(f"Testing {num_steps} consecutive training steps...")
        
        initial_params = None
        for step in range(num_steps):
            logging.info(f"\n--- Training Step {step + 1} ---")
            
            # Get current parameters
            success, params = self.test_get_params()
            if not success:
                return False
            
            if step == 0:
                initial_params = params
            
            # Perform training step
            if not self.test_train_step():
                return False
            
            time.sleep(0.5)  # Brief pause between steps
        
        # Get final parameters and compare
        logging.info("\n--- Final Parameter Comparison ---")
        success, final_params = self.test_get_params()
        if not success:
            return False
        
        if initial_params and final_params:
            logging.info(f"Initial step: {initial_params['step']}")
            logging.info(f"Final step: {final_params['step']}")
            logging.info(f"Steps increased by: {final_params['step'] - initial_params['step']}")
            
            # Check if parameters changed
            initial_logits = initial_params['mask_logits']
            final_logits = final_params['mask_logits']
            
            param_changed = any(abs(final_logits[i] - initial_logits[i]) > 1e-6 
                              for i in range(len(initial_logits)))
            
            if param_changed:
                logging.info("✅ Parameters updated during training")
                for i, (init, final) in enumerate(zip(initial_logits, final_logits)):
                    change = final - init
                    logging.info(f"  Param {i}: {init:.6f} → {final:.6f} (Δ={change:.6f})")
            else:
                logging.warning("⚠️ Parameters did not change during training")
        
        return True
    
    def run_all_tests(self, d: int = 3):
        """Run all learner tests"""
        logging.info("=== Starting Learner Tests ===")
        
        tests = [
            ("Health Check", self.test_health),
            ("Status Check", self.test_status),
            ("Parameter Retrieval", lambda: self.test_get_params(d)[0]),
            ("Single Training Step", lambda: self.test_train_step(d)),
            ("Multiple Training Steps", lambda: self.test_multiple_steps(3))
        ]
        
        results = []
        for test_name, test_func in tests:
            logging.info(f"\n--- {test_name} ---")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                logging.error(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        logging.info("\n=== Test Results ===")
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logging.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        logging.info(f"\nOverall: {passed}/{len(results)} tests passed")
        return passed == len(results)

def start_learner_server(d: int = 3, port: int = 8002):
    """Start learner server for testing"""
    cmd = [
        sys.executable, "learner_server.py",
        "--d", str(d),
        "--k", "5",  # Number of components
        "--lr", "0.001",
        "--sparsity-weight", "0.1",
        "--tau-init", "1.0",
        "--host", "localhost",
        "--port", str(port),
        "--device", "cuda:0",  # Same GPU as collector for testing
        "--log-level", "INFO"
    ]
    
    logging.info(f"Starting learner server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Learner Server")
    parser.add_argument("--d", type=int, default=3, help="Number of attributes")
    parser.add_argument("--port", type=int, default=8002, help="Learner port")
    parser.add_argument("--start-server", action="store_true", help="Start learner server automatically")
    parser.add_argument("--wait-time", type=int, default=30, help="Time to wait for server startup")
    
    args = parser.parse_args()
    
    learner_process = None
    
    try:
        if args.start_server:
            logging.info("Starting learner server...")
            learner_process = start_learner_server(args.d, args.port)
            
            # Wait for server to be ready
            logging.info(f"Waiting up to {args.wait_time} seconds for server to be ready...")
            tester = LearnerTester(f"http://localhost:{args.port}")
            
            ready = False
            for attempt in range(0, args.wait_time, 5):
                time.sleep(5)
                # Suppress health check error messages during startup
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.CRITICAL)
                
                try:
                    health_ok = tester.test_health()
                finally:
                    logging.getLogger().setLevel(original_level)
                
                if health_ok:
                    logging.info(f"✅ Server ready after ~{attempt + 5} seconds!")
                    ready = True
                    break
                elif attempt % 15 == 0:
                    logging.info(f"Still waiting for server... ({attempt + 5}/{args.wait_time}s)")
            
            if not ready:
                logging.error(f"❌ Server failed to start within {args.wait_time} seconds")
                return 1

        # Run tests
        if not args.start_server:
            tester = LearnerTester(f"http://localhost:{args.port}")
        success = tester.run_all_tests(args.d)
        
        if success:
            logging.info("🎉 All tests passed!")
        else:
            logging.error("💥 Some tests failed!")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logging.info("Tests interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Test runner error: {e}")
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
                try:
                    learner_process.wait(timeout=3.0)
                    logging.info("Learner server force killed")
                except subprocess.TimeoutExpired:
                    logging.error("Failed to kill learner server!")
            except Exception as e:
                logging.error(f"Error stopping learner server: {e}")

if __name__ == "__main__":
    sys.exit(main())