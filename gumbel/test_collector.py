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
    
    def test_generate_batch(self, d: int = 10):
        """Test batch generation endpoint"""
        logging.info("Testing batch generation...")
        try:
            # Create test request
            request_data = {
                "users_per_batch": 2,
                "samples_per_user": 3,
                "behavior_logits": [0.5] * d,  # Neutral logits
                "tau": 1.0
            }
            
            logging.info(f"Sending request: {request_data}")
            
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                json=request_data,
                timeout=60.0  # VLLM can be slow
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
                    
                    # Show sample data
                    logging.info("Sample data:")
                    if user_data.get('prompts'):
                        logging.info(f"  Sample prompt: {user_data['prompts'][0][:100]}...")
                    if user_data.get('outputs'):
                        logging.info(f"  Sample output: {user_data['outputs'][0][:100]}...")
                    if user_data.get('user_ids'):
                        logging.info(f"  Sample user_id: {user_data['user_ids'][0]}")
                    
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
    
    def run_all_tests(self, d: int = 10):
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

def start_collector_server(dataset_path: str, d: int = 10, port: int = 8001):
    """Start collector server for testing"""
    cmd = [
        sys.executable, "collector_server.py",
        "--d", str(d),
        "--dataset-path", dataset_path,
        "--vllm-model", "microsoft/DialoGPT-medium",  # Small model for testing
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
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument("--d", type=int, default=10, help="Number of attributes")
    parser.add_argument("--port", type=int, default=8001, help="Collector port")
    parser.add_argument("--start-server", action="store_true", help="Start collector server automatically")
    parser.add_argument("--wait-time", type=int, default=30, help="Time to wait for server startup")
    
    args = parser.parse_args()
    
    collector_process = None
    
    try:
        if args.start_server:
            logging.info("Starting collector server...")
            collector_process = start_collector_server(args.dataset_path, args.d, args.port)
            
            logging.info(f"Waiting {args.wait_time} seconds for server to start...")
            time.sleep(args.wait_time)
        
        # Run tests
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
                collector_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logging.warning("Force killing collector server")
                collector_process.kill()

if __name__ == "__main__":
    sys.exit(main())