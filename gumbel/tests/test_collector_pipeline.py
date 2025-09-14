#!/usr/bin/env python3
"""
Collector Pipeline Validation Script
Tests the modified collector that uses external VLLM server instead of direct model loading.
"""

import requests
import json
import time
import argparse
import sys
from typing import Dict, Any, Optional

class CollectorPipelineValidator:
    def __init__(self, collector_url: str, vllm_server_url: str):
        self.collector_url = collector_url.rstrip('/')
        self.vllm_server_url = vllm_server_url.rstrip('/')
        
    def test_vllm_server(self) -> bool:
        """Test if VLLM server is running and accessible"""
        print("🔧 Testing VLLM Server...")
        try:
            # Check models endpoint
            response = requests.get(f"{self.vllm_server_url}/models", timeout=30)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['id'] for model in models.get('data', [])]
                print(f"✅ VLLM server is running. Available models: {available_models}")
                return True
            else:
                print(f"❌ VLLM server returned HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to VLLM server at {self.vllm_server_url}")
            print("   Make sure to start VLLM server first:")
            print(f"   python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0 --port 8000")
            return False
        except Exception as e:
            print(f"❌ VLLM server test failed: {e}")
            return False
    
    def test_collector_health(self) -> bool:
        """Test collector health endpoint"""
        print("\n💚 Testing Collector Health...")
        try:
            response = requests.get(f"{self.collector_url}/health", timeout=30)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Collector health check passed: {health_data}")
                
                # Simplified health check (no detailed readiness indicators)
                print("✅ Collector health check shows healthy status")
                    
                return health_data.get('status') == 'healthy'
            else:
                print(f"❌ Collector health check failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to collector at {self.collector_url}")
            print("   Make sure collector is running with the new API-based configuration")
            return False
        except Exception as e:
            print(f"❌ Collector health test failed: {e}")
            return False
    
    def test_collector_status(self) -> bool:
        """Test collector status endpoint"""
        print("\n📊 Testing Collector Status...")
        try:
            response = requests.get(f"{self.collector_url}/status", timeout=30)
            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ Collector status: {status_data}")
                return status_data.get('status') == 'running'
            else:
                print(f"❌ Collector status failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Collector status test failed: {e}")
            return False
    
    def test_vllm_api_call(self) -> bool:
        """Test direct VLLM API call"""
        print("\n🤖 Testing VLLM API Call...")
        try:
            # Test completion endpoint
            response = requests.post(
                f"{self.vllm_server_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 20,
                    "temperature": 0.0
                },
                timeout=60
            )
            
            if response.status_code == 200:
                completion = response.json()
                message = completion['choices'][0]['message']['content']
                print(f"✅ VLLM API call successful. Response: '{message[:50]}...'")
                return True
            else:
                print(f"❌ VLLM API call failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ VLLM API call failed: {e}")
            return False
    
    def test_batch_generation(self) -> bool:
        """Test collector batch generation (the main pipeline)"""
        print("\n🎲 Testing Collector Batch Generation...")
        try:
            # Prepare test request (small batch) - simplified API
            test_request = {
                "users_per_batch": 1,
                "samples_per_user": 1
            }
            
            print(f"   Requesting: {test_request['users_per_batch']} users × {test_request['samples_per_user']} samples")
            print("   This tests the full pipeline: data sampling → VLLM API calls → reward computation")
            
            start_time = time.time()
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                headers={"Content-Type": "application/json"},
                json=test_request,
                timeout=300  # 5 minutes max
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                batch_data = response.json()
                
                if batch_data.get('success'):
                    print(f"✅ Batch generation successful in {elapsed_time:.2f} seconds")
                    
                    R = batch_data.get('R', [])
                    if R:
                        print(f"   Reward matrix: {len(R)} samples × {len(R[0])} attributes")
                        reward_range = [min(min(row) for row in R), max(max(row) for row in R)]
                        print(f"   Reward range: [{reward_range[0]:.3f}, {reward_range[1]:.3f}]")
                    
                    user_data = batch_data.get('user_data', {})
                    if user_data:
                        prompts = user_data.get('prompts', [])
                        outputs = user_data.get('outputs', [])
                        print(f"   Generated {len(prompts)} prompts and {len(outputs)} outputs")
                        
                        if prompts and outputs:
                            print(f"   Sample prompt: '{prompts[0][:50]}...'")
                            print(f"   Sample output: '{outputs[0][:50]}...'")
                    
                    print(f"\n🎉 SUCCESS: Pipeline is working with API-based architecture!")
                    print(f"   Performance: {elapsed_time:.2f}s for {test_request['users_per_batch']}×{test_request['samples_per_user']} samples")
                    return True
                else:
                    error = batch_data.get('error', 'Unknown error')
                    print(f"❌ Batch generation failed: {error}")
                    print(f"   Full response: {batch_data}")
                    return False
            else:
                print(f"❌ Batch generation HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"❌ Batch generation timed out (>300s)")
            print("   This might indicate the API calls are hanging")
            return False
        except Exception as e:
            print(f"❌ Batch generation test failed: {e}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete pipeline validation"""
        print("🚀 Collector Pipeline Validation")
        print("=" * 50)
        print("Testing the modified collector that uses external VLLM server")
        print()
        
        all_passed = True
        
        # Test 1: VLLM Server
        if not self.test_vllm_server():
            all_passed = False
            print("\n❌ Cannot proceed without VLLM server")
            return False
        
        # Test 2: Collector Health  
        if not self.test_collector_health():
            all_passed = False
            print("\n❌ Collector health check failed")
            return False
        
        # Test 3: Collector Status
        if not self.test_collector_status():
            all_passed = False
        
        # Test 4: Direct VLLM API
        if not self.test_vllm_api_call():
            all_passed = False
            print("\n⚠️  VLLM API test failed, but continuing...")
        
        # Test 5: Full Pipeline
        if not self.test_batch_generation():
            all_passed = False
            print("\n❌ Main pipeline test failed")
            return False
        
        # Summary
        print("\n🏁 VALIDATION SUMMARY")
        print("=" * 50)
        
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Collector successfully migrated to API-based architecture")
            print("✅ Performance should be significantly improved")
            print("\nYou can now run the full coordinator:")
            print("  python coordinator.py --max-steps 1000")
        else:
            print("💥 Some tests failed.")
            print("   Check the error messages above and fix issues before proceeding.")
            
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Validate collector pipeline with external VLLM server")
    parser.add_argument("--collector-url", type=str, default="http://localhost:8001", 
                       help="Collector server URL")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000/v1",
                       help="VLLM server URL")
    parser.add_argument("--collector-host", type=str, help="Collector hostname (for cluster)")
    parser.add_argument("--vllm-host", type=str, help="VLLM server hostname (for cluster)")
    
    args = parser.parse_args()
    
    # Handle cluster hostnames
    collector_url = args.collector_url
    vllm_server_url = args.vllm_server_url
    
    if args.collector_host:
        collector_url = f"http://{args.collector_host}:8001"
    if args.vllm_host:
        vllm_server_url = f"http://{args.vllm_host}:8000/v1"
    
    print(f"Collector URL: {collector_url}")
    print(f"VLLM Server URL: {vllm_server_url}")
    print()
    
    validator = CollectorPipelineValidator(collector_url, vllm_server_url)
    success = validator.run_full_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()