#!/usr/bin/env python3
"""
Test script for collector server to verify it's working properly.
Tests data sampling, VLLM scoring, and API endpoints.
"""

import requests
import time
import logging
import json
import torch
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CollectorTester:
    def __init__(self, collector_url: str = "http://localhost:8001"):
        self.collector_url = collector_url.rstrip('/')
        
    def test_health(self):
        """Test health endpoint"""
        logging.info("Testing health endpoint...")
        try:
            response = requests.get(f"{self.collector_url}/health", timeout=5.0)
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
        logging.info("Testing status endpoint...")
        try:
            response = requests.get(f"{self.collector_url}/status", timeout=5.0)
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
    
    def test_generate_batch(self, d: int = 3):
        """Test batch generation endpoint with focus on reward logic"""
        logging.info("Testing batch generation and reward logic...")
        try:
            # Create test request - use small numbers for focused testing
            request_data = {
                "users_per_batch": 1,  # Just 1 user
                "samples_per_user": 3,  # 3 samples (all from our test data)
                "behavior_logits": [0.8, 0.2, 0.6],  # Non-uniform logits for d=3
                "tau": 1.0
            }
            
            logging.info(f"Sending request: {request_data}")
            
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                json=request_data,
                timeout=180.0  # VLLM scoring can be very slow, especially first time
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["success"]:
                    logging.info("✅ Batch generation successful!")
                    
                    # Check response structure
                    m_hard = result["m_hard"]
                    R = result["R"]
                    user_data = result["user_data"]
                    
                    logging.info(f"  m_hard shape: {len(m_hard)} (expected: {d})")
                    logging.info(f"  R shape: {len(R)}x{len(R[0]) if R else 0} (expected: 6x{d})")
                    logging.info(f"  user_data keys: {list(user_data.keys())}")
                    logging.info(f"  prompts count: {len(user_data.get('prompts', []))}")
                    logging.info(f"  outputs count: {len(user_data.get('outputs', []))}")
                    logging.info(f"  user_ids count: {len(user_data.get('user_ids', []))}")
                    
                    # Check dimensions
                    expected_batch_size = request_data["users_per_batch"] * request_data["samples_per_user"]
                    actual_batch_size = len(R)
                    
                    if actual_batch_size == expected_batch_size:
                        logging.info(f"✅ Batch size correct: {actual_batch_size}")
                    else:
                        logging.warning(f"⚠️ Batch size mismatch: expected {expected_batch_size}, got {actual_batch_size}")
                    
                    if len(m_hard) == d:
                        logging.info(f"✅ Mask dimension correct: {d}")
                    else:
                        logging.warning(f"⚠️ Mask dimension mismatch: expected {d}, got {len(m_hard)}")
                    
                    if R and len(R[0]) == d:
                        logging.info(f"✅ Reward matrix dimension correct: {len(R[0])}")
                    else:
                        logging.warning(f"⚠️ Reward matrix dimension mismatch: expected {d}, got {len(R[0]) if R else 0}")
                    
                    # Expected behavior explanation
                    logging.info("=== EXPECTED BEHAVIOR ===")
                    logging.info("Attribute 0: Pirate persona - should give highest reward to response 0 (pirate talk)")
                    logging.info("Attribute 1: Academic persona - should give highest reward to response 1 (formal language)")  
                    logging.info("Attribute 2: Teen persona - should give highest reward to response 2 (slang/emojis)")
                    logging.info("All responses are about the same topic, so differences should be due to style/persona matching")
                    
                    # Detailed reward analysis for testing
                    logging.info("\n=== REWARD ANALYSIS ===")
                    logging.info(f"Hard mask: {m_hard}")
                    logging.info(f"Active attributes: {sum(m_hard)}")
                    
                    if R:
                        import numpy as np
                        R_array = np.array(R)
                        logging.info(f"Reward matrix shape: {R_array.shape}")
                        logging.info(f"Reward range: [{R_array.min():.4f}, {R_array.max():.4f}]")
                        logging.info(f"Mean reward per attribute: {R_array.mean(axis=0)}")
                        
                        # Show rewards for each sample with interpretation
                        personas = ["Pirate", "Academic", "Teen"]
                        response_styles = ["Pirate style", "Academic style", "Teen style"]
                        
                        for i, row in enumerate(R):
                            logging.info(f"  Response {i} ({response_styles[i]}): {[f'{r:.4f}' for r in row]}")
                            
                        # Highlight expected patterns
                        logging.info("\n=== REWARD INTERPRETATION ===")
                        for attr_idx, persona in enumerate(personas):
                            attr_rewards = [R[i][attr_idx] for i in range(len(R))]
                            best_response = attr_rewards.index(max(attr_rewards))
                            logging.info(f"{persona} attribute (col {attr_idx}): highest reward for response {best_response} ({response_styles[best_response]})")
                            if best_response == attr_idx:
                                logging.info("  ✅ CORRECT: Attribute gave highest reward to matching response style!")
                            else:
                                logging.info("  ❌ UNEXPECTED: Attribute gave highest reward to non-matching style")
                    
                    # Show sample data
                    logging.info("\n=== SAMPLE DATA ===")
                    if user_data.get('prompts'):
                        for i, prompt in enumerate(user_data['prompts']):
                            logging.info(f"  Prompt {i}: {prompt}")
                    if user_data.get('outputs'):
                        for i, output in enumerate(user_data['outputs']):
                            logging.info(f"  Response {i}: {output[:100]}...")
                    if user_data.get('user_ids'):
                        logging.info(f"  User IDs: {user_data['user_ids']}")
                    
                    return True
                else:
                    logging.error(f"❌ Batch generation failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                logging.error(f"❌ Batch generation HTTP error: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Batch generation error: {e}")
            return False
    
    def run_all_tests(self, d: int = 3):
        """Run all tests"""
        logging.info("=== Starting Collector Tests ===")
        
        tests = [
            ("Health Check", self.test_health),
            ("Status Check", self.test_status),
            ("Batch Generation", lambda: self.test_generate_batch(d))
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

def start_collector_server(dataset_path: str, attribute_prompts_path: str, d: int = 3, port: int = 8001):
    """Start collector server for testing"""
    cmd = [
        sys.executable, "collector_server.py",
        "--d", str(d),
        "--dataset-path", dataset_path,
        "--attribute-prompts-path", attribute_prompts_path,
        "--vllm-model", "meta-llama/Llama-3.2-1B-Instruct",  # Small model for testing
        "--gpu-memory-util", "0.3",  # Low memory usage
        "--host", "localhost",
        "--port", str(port),
        "--device", "cuda:0",
        "--log-level", "INFO"
    ]
    
    logging.info(f"Starting collector server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Collector Server")
    parser.add_argument("--dataset-path", type=str, default="test_data.pkl", help="Dataset path")
    parser.add_argument("--attribute-prompts-path", type=str, default="test_attribute_prompts.json", help="Path to attribute prompts JSON file")
    parser.add_argument("--d", type=int, default=3, help="Number of attributes")
    parser.add_argument("--port", type=int, default=8001, help="Collector port")
    parser.add_argument("--start-server", action="store_true", help="Start collector server automatically")
    parser.add_argument("--wait-time", type=int, default=30, help="Time to wait for server startup")
    
    args = parser.parse_args()
    
    collector_process = None
    
    try:
        if args.start_server:
            logging.info("Starting collector server...")
            collector_process = start_collector_server(args.dataset_path, args.attribute_prompts_path, args.d, args.port)
            
            # Wait for server to be ready with health checks
            logging.info(f"Waiting up to {args.wait_time} seconds for server to be ready...")
            tester = CollectorTester(f"http://localhost:{args.port}")
            
            ready = False
            for attempt in range(0, args.wait_time, 5):  # Check every 5 seconds
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
                elif attempt % 30 == 0:  # Log progress every 30 seconds
                    logging.info(f"Still waiting for server... ({attempt + 5}/{args.wait_time}s)")
            
            if not ready:
                logging.error(f"❌ Server failed to start within {args.wait_time} seconds")
                return 1

        # Run tests (reuse tester if we created one above)
        if not args.start_server:
            tester = CollectorTester(f"http://localhost:{args.port}")
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
        if collector_process:
            logging.info("Stopping collector server...")
            collector_process.terminate()
            try:
                collector_process.wait(timeout=5.0)
                logging.info("Collector server stopped gracefully")
            except subprocess.TimeoutExpired:
                logging.warning("Force killing collector server")
                collector_process.kill()
                try:
                    collector_process.wait(timeout=3.0)
                    logging.info("Collector server force killed")
                except subprocess.TimeoutExpired:
                    logging.error("Failed to kill collector server!")
            except Exception as e:
                logging.error(f"Error stopping collector server: {e}")

if __name__ == "__main__":
    sys.exit(main())