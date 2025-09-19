#!/usr/bin/env python3

import asyncio
from literegistry import FileSystemKVStore, RegistryClient

async def debug_bandit():
    """Debug the bandit algorithm to see what servers it's selecting"""
    
    registry_path = "/gscratch/ark/devinl6/registry"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    fileSystemKVStore = FileSystemKVStore(registry_path)
    registryClient = RegistryClient(fileSystemKVStore, service_type="model_path")
    
    # Get all servers
    all_servers = await registryClient.get_all(model_name)
    print(f"All servers: {all_servers}")
    
    # Test what sample_servers returns
    print(f"\nTesting sample_servers with different n values:")
    
    for n in [1, 5, 10, 512]:
        servers = await registryClient.sample_servers(model_name, n=n)
        print(f"n={n}: {servers}")
        
        # Check if all servers are the same
        unique_servers = set(servers)
        print(f"  Unique servers: {len(unique_servers)} out of {len(servers)}")
        if len(unique_servers) == 1:
            print(f"  ⚠️  All {len(servers)} requests would go to: {list(unique_servers)[0]}")
        else:
            print(f"  ✅ Good distribution across {len(unique_servers)} servers")

if __name__ == "__main__":
    asyncio.run(debug_bandit())
