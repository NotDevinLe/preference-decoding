#!/usr/bin/env python3

import asyncio
import logging
import json
from typing import List, Dict, Any
from literegistry import RegistryHTTPClient, FileSystemKVStore, RegistryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def test_literegistry():
    """Simple test script to verify LiteRegistry is working"""
    
    # Initialize registry components
    registry_path = "/gscratch/ark/devinl6/registry"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Initializing LiteRegistry with path: {registry_path}")
    fileSystemKVStore = FileSystemKVStore(registry_path)
    registryClient = RegistryClient(fileSystemKVStore, service_type="model_path")
    
    # Check what servers are available
    print(f"\nChecking available servers for model: {model_name}")
    available_servers = await registryClient.get_all(model_name)
    
    if not available_servers:
        print(f"❌ No servers found for model {model_name}")
        print("Make sure you have vLLM servers running and registered!")
        return
    
    print(f"✅ Found {len(available_servers)} servers:")
    for i, server in enumerate(available_servers):
        print(f"  {i+1}. {server}")
    
    # Test connectivity to each server
    print(f"\nTesting server connectivity...")
    healthy_servers = []
    
    for i, server in enumerate(available_servers):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as test_session:
                async with test_session.get(f"{server}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"✅ Server {i+1}: {server} - HEALTHY")
                        healthy_servers.append(server)
                    else:
                        print(f"❌ Server {i+1}: {server} - UNHEALTHY (status {resp.status})")
        except Exception as e:
            print(f"❌ Server {i+1}: {server} - UNREACHABLE ({type(e).__name__}: {e})")
    
    if not healthy_servers:
        print("\n❌ No healthy servers found! Cannot proceed with test.")
        return
    
    print(f"\n✅ {len(healthy_servers)} healthy servers available for testing")
    
    # Test simple request using RegistryHTTPClient
    print(f"\nTesting request with RegistryHTTPClient...")
    
    test_payload = {
        "model": model_name,
        "prompt": "Hello, how are you?",
        "max_tokens": 10,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1
    }
    
    try:
        async with RegistryHTTPClient(
            registry=registryClient,
            value=model_name,
            max_parallel_requests=512,
            timeout=30,
            max_retries=3
        ) as httpClient:
            print("Making test request...")
            result, server_used = await httpClient.request_with_rotation("v1/completions", test_payload)
            print(f"✅ Request successful!")
            print(f"   Server used: {server_used}")
            print(f"   Response keys: {list(result.keys())}")
            if "choices" in result and result["choices"]:
                print(f"   Generated text: {result['choices'][0].get('text', 'N/A')[:100]}...")
    except Exception as e:
        print(f"❌ Request failed: {type(e).__name__}: {e}")
        return
    
    # Test server rotation with 100,000 requests
    print(f"\n🚀 Starting 100,000 request stress test...")
    
    test_payload = {
        "model": model_name,
        "prompt": "Hello, this is a test request",
        "max_tokens": 5,
        "temperature": 0.0
    }
    
    try:
        async with RegistryHTTPClient(
            registry=registryClient,
            value=model_name,
            max_parallel_requests=512,
            timeout=30,
            max_retries=3
        ) as httpClient:
            servers_used = []
            request_count = 0
            success_count = 0
            failure_count = 0
            
            print("Starting spam test... (Press Ctrl+C to stop early)")
            
            # Send all 100,000 requests at once!
            total_requests = 100000
            print(f"Creating {total_requests} requests...")
            
            # Create all 100,000 request tasks at once
            all_tasks = []
            for i in range(total_requests):
                task = httpClient.request_with_rotation("v1/completions", test_payload)
                all_tasks.append(task)
            
            print(f"🚀 Launching all {total_requests} requests simultaneously...")
            start_time = asyncio.get_event_loop().time()
            
            # Execute all requests in parallel
            try:
                all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                
                end_time = asyncio.get_event_loop().time()
                total_time = end_time - start_time
                
                print(f"⚡ All requests completed in {total_time:.2f} seconds!")
                print(f"   Rate: {total_requests/total_time:.1f} requests/second")
                
                # Process all results
                for i, result in enumerate(all_results):
                    if isinstance(result, Exception):
                        failure_count += 1
                        if i < 10:  # Only print first 10 failures
                            print(f"Request {i + 1} failed: {result}")
                    else:
                        result_data, server_used = result
                        servers_used.append(server_used)
                        success_count += 1
                    
                    request_count += 1
                
            except Exception as e:
                print(f"❌ All requests failed: {e}")
                return
            
            print(f"\n📊 Final Results:")
            print(f"  Total requests: {request_count}")
            print(f"  Successful: {success_count}")
            print(f"  Failed: {failure_count}")
            print(f"  Success rate: {(success_count/request_count)*100:.1f}%")
            
            # Analyze server usage
            from collections import Counter
            server_counts = Counter(servers_used)
            print(f"\n📊 Server usage distribution:")
            for server, count in server_counts.items():
                percentage = (count/len(servers_used))*100 if servers_used else 0
                print(f"  {server}: {count} requests ({percentage:.1f}%)")
            
            if len(set(servers_used)) > 1:
                print("✅ Server rotation is working!")
            else:
                print("⚠️  All requests used the same server - rotation may not be working properly")
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user at request {request_count}")
        print(f"  Successful: {success_count}, Failed: {failure_count}")
    except Exception as e:
        print(f"❌ Stress test failed: {type(e).__name__}: {e}")
        return
    
    print(f"\n🎉 LiteRegistry test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_literegistry())
