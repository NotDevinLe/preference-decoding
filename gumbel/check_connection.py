#!/usr/bin/env python3
import requests
import argparse
import json

def check_server(url, name="Server"):
    """Check health and status of a server"""
    print(f"🔍 Checking {name} at {url}")
    
    # Health check
    try:
        health_resp = requests.get(f"{url}/health", timeout=10)
        print(f"  Health: {health_resp.status_code}")
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            print(f"  Health data: {health_data}")
        else:
            print(f"  Health response: {health_resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"  Health error: {e}")
        return False
    
    # Status check
    try:
        status_resp = requests.get(f"{url}/status", timeout=10)
        print(f"  Status: {status_resp.status_code}")
        if status_resp.status_code == 200:
            status_data = status_resp.json()
            print(f"  Status data: {json.dumps(status_data, indent=4)}")
        else:
            print(f"  Status response: {status_resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"  Status error: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Check server connections")
    parser.add_argument("--node", type=str, default="g3097", help="Node name")
    parser.add_argument("--collector-port", type=int, default=8001, help="Collector port")
    parser.add_argument("--learner-port", type=int, default=8002, help="Learner port")
    parser.add_argument("--server", type=str, choices=['collector', 'learner', 'both'], 
                       default='both', help="Which server to check")
    parser.add_argument("--url", type=str, help="Direct URL to check (overrides node/port)")
    
    args = parser.parse_args()
    
    if args.url:
        check_server(args.url, "Custom Server")
    else:
        if args.server in ['collector', 'both']:
            collector_url = f"http://{args.node}:{args.collector_port}"
            print("=" * 50)
            if not check_server(collector_url, "Collector"):
                print("❌ Collector check failed")
            print()
        
        if args.server in ['learner', 'both']:
            learner_url = f"http://{args.node}:{args.learner_port}"
            print("=" * 50)
            if not check_server(learner_url, "Learner"):
                print("❌ Learner check failed")
            print()
    
    print("✅ Connection checks complete")

if __name__ == "__main__":
    main()