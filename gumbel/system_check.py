#!/usr/bin/env python3
"""
Advanced system checker for distributed sparse attribute learning.
Verifies collector and learner servers are running and can communicate.
"""
import requests
import time
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from load_config import load_config, get_coordinator_args, print_config_summary

class SystemChecker:
    def __init__(self, collector_url: str, learner_url: str, timeout: float = 10.0):
        self.collector_url = collector_url.rstrip('/')
        self.learner_url = learner_url.rstrip('/')
        self.timeout = timeout
        
    def check_server_health(self, url: str, name: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if server is healthy and responsive"""
        try:
            print(f"🔍 Checking {name} at {url}/health...")
            response = requests.get(f"{url}/health", timeout=self.timeout)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ {name} is healthy: {health_data}")
                return True, health_data
            else:
                print(f"❌ {name} returned HTTP {response.status_code}")
                return False, {"error": f"HTTP {response.status_code}", "text": response.text}
                
        except requests.exceptions.ConnectTimeout:
            print(f"❌ {name} connection timeout (server may not be running)")
            return False, {"error": "connection_timeout"}
        except requests.exceptions.ConnectionError:
            print(f"❌ {name} connection refused (server not running or wrong port)")
            return False, {"error": "connection_refused"}
        except requests.exceptions.Timeout:
            print(f"❌ {name} request timeout")
            return False, {"error": "request_timeout"}
        except Exception as e:
            print(f"❌ {name} error: {e}")
            return False, {"error": str(e)}
    
    def check_server_status(self, url: str, name: str) -> Tuple[bool, Dict[str, Any]]:
        """Get detailed server status"""
        try:
            print(f"📊 Getting {name} status...")
            response = requests.get(f"{url}/status", timeout=self.timeout)
            
            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ {name} status:")
                for key, value in status_data.items():
                    print(f"   {key}: {value}")
                return True, status_data
            else:
                print(f"❌ {name} status failed: HTTP {response.status_code}")
                return False, {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ {name} status error: {e}")
            return False, {"error": str(e)}
    
    def check_learner_params(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if learner can provide parameters"""
        try:
            print(f"🧠 Getting learner parameters...")
            response = requests.get(f"{self.learner_url}/get_params", timeout=self.timeout)
            
            if response.status_code == 200:
                params_data = response.json()
                if params_data.get("success"):
                    print(f"✅ Learner parameters available:")
                    print(f"   Step: {params_data.get('step', 0)}")
                    print(f"   Temperature: {params_data.get('tau', 1.0):.4f}")
                    mask_logits = params_data.get('mask_logits', [])
                    if mask_logits:
                        print(f"   Mask logits: {len(mask_logits)} values, range [{min(mask_logits):.4f}, {max(mask_logits):.4f}]")
                    return True, params_data
                else:
                    error = params_data.get("error", "Unknown error")
                    print(f"❌ Learner parameters failed: {error}")
                    return False, params_data
            else:
                print(f"❌ Learner parameters HTTP {response.status_code}")
                return False, {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Learner parameters error: {e}")
            return False, {"error": str(e)}
    
    def test_collector_batch_generation(self) -> Tuple[bool, Dict[str, Any]]:
        """Test collector can generate batches"""
        try:
            print(f"🎲 Testing collector batch generation...")
            
            # First get learner params for behavior policy
            learner_ok, params_data = self.check_learner_params()
            if not learner_ok or not params_data.get("success"):
                print("❌ Cannot test collector - learner parameters not available")
                return False, {"error": "learner_params_unavailable"}
            
            # Test batch generation
            test_request = {
                "users_per_batch": 2,
                "samples_per_user": 2, 
                "behavior_logits": params_data["mask_logits"],
                "tau": params_data["tau"]
            }
            
            print(f"   Requesting batch: {test_request['users_per_batch']} users × {test_request['samples_per_user']} samples")
            response = requests.post(
                f"{self.collector_url}/generate_batch",
                json=test_request,
                timeout=max(30.0, self.timeout)  # VLLM can be slow
            )
            
            if response.status_code == 200:
                batch_data = response.json()
                if batch_data.get("success"):
                    print(f"✅ Collector batch generation successful:")
                    print(f"   Hard mask sparsity: {sum(batch_data.get('m_hard', []))}")
                    R = batch_data.get('R', [])
                    if R:
                        print(f"   Reward matrix: {len(R)} samples × {len(R[0])} attributes")
                        print(f"   Reward range: [{min(min(row) for row in R):.3f}, {max(max(row) for row in R):.3f}]")
                    user_data = batch_data.get('user_data', {})
                    if user_data:
                        print(f"   Generated {len(user_data.get('prompts', []))} prompts and outputs")
                    return True, batch_data
                else:
                    error = batch_data.get("error", "Unknown error")
                    print(f"❌ Collector batch generation failed: {error}")
                    return False, batch_data
            else:
                print(f"❌ Collector batch generation HTTP {response.status_code}")
                return False, {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Collector batch generation error: {e}")
            return False, {"error": str(e)}
    
    def test_learner_training_step(self, batch_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Test learner can perform training step"""
        try:
            print(f"🎯 Testing learner training step...")
            
            # Prepare training request
            train_request = {
                "m_hard": batch_data["m_hard"],
                "R": batch_data["R"],
                "user_data": batch_data["user_data"],
                "success": True
            }
            
            response = requests.post(
                f"{self.learner_url}/train_step",
                json=train_request,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✅ Learner training step successful:")
                    print(f"   Step: {result.get('step', 0)}")
                    print(f"   Loss: {result.get('loss', 0.0):.4f}")
                    print(f"   Reward signal: {result.get('reward_signal', 0.0):.4f}")
                    print(f"   Active attributes: {result.get('active_attributes', 0.0):.1f}")
                    return True, result
                else:
                    error = result.get("error", "Unknown error")
                    print(f"❌ Learner training step failed: {error}")
                    return False, result
            else:
                print(f"❌ Learner training step HTTP {response.status_code}")
                return False, {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Learner training step error: {e}")
            return False, {"error": str(e)}
    
    def run_full_system_check(self) -> bool:
        """Run complete system verification"""
        print("🚀 Starting Full System Check")
        print("=" * 50)
        
        all_passed = True
        
        # 1. Check server health
        print("\n1️⃣ HEALTH CHECKS")
        print("-" * 20)
        
        collector_healthy, collector_health = self.check_server_health(self.collector_url, "Collector")
        if not collector_healthy:
            all_passed = False
            
        learner_healthy, learner_health = self.check_server_health(self.learner_url, "Learner")
        if not learner_healthy:
            all_passed = False
            
        if not (collector_healthy and learner_healthy):
            print("\n❌ Health checks failed. Cannot proceed with integration tests.")
            return False
        
        # 2. Check server status
        print("\n2️⃣ STATUS CHECKS")
        print("-" * 20)
        
        collector_status_ok, collector_status = self.check_server_status(self.collector_url, "Collector")
        learner_status_ok, learner_status = self.check_server_status(self.learner_url, "Learner")
        
        if not (collector_status_ok and learner_status_ok):
            all_passed = False
        
        # 3. Test collector batch generation
        print("\n3️⃣ INTEGRATION TESTS")
        print("-" * 20)
        
        batch_ok, batch_data = self.test_collector_batch_generation()
        if not batch_ok:
            all_passed = False
            print("\n❌ Cannot test learner training - collector batch generation failed")
        else:
            # 4. Test learner training step
            train_ok, train_result = self.test_learner_training_step(batch_data)
            if not train_ok:
                all_passed = False
        
        # Summary
        print("\n🏁 SYSTEM CHECK SUMMARY")
        print("=" * 50)
        
        if all_passed:
            print("🎉 All checks passed! System is ready for distributed training.")
            print("\nYou can now run:")
            print("  python coordinator.py --enable-wandb-coordinator")
        else:
            print("💥 Some checks failed. Please fix issues before running coordinator.")
            
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="System health check for distributed training")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file")
    parser.add_argument("--collector-url", type=str, help="Collector URL override")
    parser.add_argument("--learner-url", type=str, help="Learner URL override")
    parser.add_argument("--node", type=str, help="Node name (for both servers)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument("--quick", action="store_true", help="Quick health check only")
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        coordinator_config = get_coordinator_args(config)
        
        if not args.quick:
            print_config_summary(config)
            print()
        
        # Determine URLs
        if args.collector_url and args.learner_url:
            collector_url = args.collector_url
            learner_url = args.learner_url
        elif args.node:
            # Both servers on same node
            collector_url = f"http://{args.node}:8001"
            learner_url = f"http://{args.node}:8002"
        else:
            # Use config URLs
            collector_url = coordinator_config['collector_url']
            learner_url = coordinator_config['learner_url']
            
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)
    
    print(f"Checking system with:")
    print(f"  Collector: {collector_url}")
    print(f"  Learner: {learner_url}")
    print(f"  Timeout: {args.timeout}s")
    print()
    
    # Create checker and run tests
    checker = SystemChecker(collector_url, learner_url, args.timeout)
    
    if args.quick:
        # Quick health check only
        collector_ok, _ = checker.check_server_health(collector_url, "Collector")
        learner_ok, _ = checker.check_server_health(learner_url, "Learner")
        success = collector_ok and learner_ok
    else:
        # Full system check
        success = checker.run_full_system_check()
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()