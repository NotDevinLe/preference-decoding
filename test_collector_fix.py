#!/usr/bin/env python3
"""
Quick test to verify the collector server fixes work.
"""

import subprocess
import sys
import time
import requests

def test_collector_startup():
    """Test that collector server starts without the async event loop error"""
    print("Testing collector server startup...")
    
    # Start the collector in background with minimal config
    cmd = [
        sys.executable, "-m", "gumbel.core.collector_server",
        "--d", "10",  # Small number for quick test
        "--dataset-path", "gumbel/data/persona_train_dataset.pkl",
        "--model-name", "meta-llama/Llama-3.2-1B-Instruct", 
        "--vllm-server-url", "http://localhost:8000",  # Won't connect but shouldn't cause startup error
        "--attribute-prompts-path", "gumbel/configs/attribute_prompts.json",
        "--port", "8099",  # Use different port for testing
        "--log-level", "DEBUG"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Start process
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a few seconds to start
        time.sleep(5)
        
        # Check if it's still running (no immediate crash)
        if proc.poll() is None:
            print("✅ Server started successfully (no immediate crash)")
            
            # Try to hit health endpoint
            try:
                response = requests.get("http://localhost:8099/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Health endpoint responded successfully")
                    print(f"Response: {response.json()}")
                else:
                    print(f"⚠️  Health endpoint returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Could not connect to health endpoint: {e}")
            
            # Terminate the server
            proc.terminate()
            proc.wait(timeout=10)
            print("✅ Server terminated cleanly")
            
        else:
            # Process exited, check for errors
            stdout, stderr = proc.communicate()
            print("❌ Server crashed on startup")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing collector: {e}")
        return False
    
    print("\n🎉 Collector server startup test passed!")
    return True

if __name__ == "__main__":
    success = test_collector_startup()
    sys.exit(0 if success else 1)