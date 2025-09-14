"""
Test script for collector server performance.
Continuously generates batches and measures throughput.
"""

import asyncio
import aiohttp
import time
import argparse
import json
import logging
from typing import Dict, Any, List
from statistics import mean, stdev
import signal
import sys


class CollectorTester:
    def __init__(self, collector_url: str = "http://localhost:8000"):
        self.collector_url = collector_url
        self.session = None
        self.running = True
        self.stats = {
            'total_batches': 0,
            'total_samples': 0,
            'batch_times': [],
            'batch_sizes': [],
            'start_time': None
        }
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n=� Received signal {signum}, shutting down...")
        self.running = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for large batches
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_collector_health(self) -> bool:
        """Check if collector server is healthy"""
        try:
            async with self.session.get(f"{self.collector_url}/health") as response:
                if response.status == 200:
                    print(" Collector server is healthy")
                    return True
                else:
                    print(f"L Collector health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"L Failed to connect to collector: {e}")
            return False
    
    async def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status"""
        try:
            async with self.session.get(f"{self.collector_url}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"=� Collector status: {status['status']}, Collections served: {status['collections_served']}")
                    return status
                else:
                    print(f"L Status check failed: {response.status}")
                    return {}
        except Exception as e:
            print(f"L Failed to get collector status: {e}")
            return {}
    
    async def generate_batch(self, users_per_batch: int, samples_per_user: int) -> Dict[str, Any]:
        """Generate a single batch from the collector"""
        payload = {
            "users_per_batch": users_per_batch,
            "samples_per_user": samples_per_user
        }
        
        batch_start = time.time()
        try:
            async with self.session.post(f"{self.collector_url}/generate_batch", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                batch_time = time.time() - batch_start
                
                if result.get('success', False):
                    batch_size = len(result.get('R', []))
                    d = len(result['R'][0]) if result.get('R') and len(result['R']) > 0 else 0
                    
                    return {
                        'success': True,
                        'batch_time': batch_time,
                        'batch_size': batch_size,
                        'dimensions': d,
                        'total_samples': users_per_batch * samples_per_user,
                        'samples_per_second': (users_per_batch * samples_per_user) / batch_time,
                        'user_data_keys': list(result.get('user_data', {}).keys())
                    }
                else:
                    print(f"L Batch generation failed: {result.get('error', 'Unknown error')}")
                    return {'success': False, 'error': result.get('error')}
                    
        except Exception as e:
            batch_time = time.time() - batch_start
            print(f"L Exception during batch generation: {e}")
            return {'success': False, 'error': str(e), 'batch_time': batch_time}
    
    def update_stats(self, batch_result: Dict[str, Any]):
        """Update performance statistics"""
        if batch_result['success']:
            self.stats['total_batches'] += 1
            self.stats['total_samples'] += batch_result['total_samples']
            self.stats['batch_times'].append(batch_result['batch_time'])
            self.stats['batch_sizes'].append(batch_result['batch_size'])
    
    def print_stats(self, batch_result: Dict[str, Any]):
        """Print current batch and cumulative statistics"""
        if not batch_result['success']:
            print(f"L Failed batch: {batch_result.get('error', 'Unknown error')}")
            return
        
        # Current batch stats
        print(f" Batch {self.stats['total_batches']:4d} | "
              f"Size: {batch_result['batch_size']:3d} | "
              f"Dims: {batch_result['dimensions']:3d} | "
              f"Time: {batch_result['batch_time']:6.2f}s | "
              f"Samples/sec: {batch_result['samples_per_second']:6.1f}")
        
        # Cumulative stats every 10 batches
        if self.stats['total_batches'] % 10 == 0:
            elapsed = time.time() - self.stats['start_time']
            avg_batch_time = mean(self.stats['batch_times'])
            avg_samples_per_sec = self.stats['total_samples'] / elapsed
            
            print(f"\n=� CUMULATIVE STATS (after {self.stats['total_batches']} batches):")
            print(f"   Total runtime: {elapsed:.1f}s")
            print(f"   Total samples: {self.stats['total_samples']}")
            print(f"   Avg batch time: {avg_batch_time:.2f}s � {stdev(self.stats['batch_times']):.2f}s")
            print(f"   Avg samples/sec: {avg_samples_per_sec:.1f}")
            print(f"   Batches/min: {self.stats['total_batches'] * 60 / elapsed:.1f}")
            
            if len(self.stats['batch_times']) > 1:
                min_time = min(self.stats['batch_times'])
                max_time = max(self.stats['batch_times'])
                print(f"   Batch time range: {min_time:.2f}s - {max_time:.2f}s")
            print()
    
    async def run_continuous_test(self, users_per_batch: int, samples_per_user: int, 
                                max_batches: int = None, batch_interval: float = 0.0):
        """Run continuous batch generation test"""
        print(f"=� Starting continuous collector test...")
        print(f"   Collector URL: {self.collector_url}")
        print(f"   Batch config: {users_per_batch} users � {samples_per_user} samples = {users_per_batch * samples_per_user} total samples per batch")
        print(f"   Max batches: {max_batches if max_batches else 'unlimited'}")
        print(f"   Batch interval: {batch_interval}s")
        print(f"   Press Ctrl+C to stop gracefully\n")
        
        # Check collector health first
        if not await self.check_collector_health():
            return
        
        await self.get_collector_status()
        print()
        
        self.stats['start_time'] = time.time()
        batch_count = 0
        
        try:
            while self.running:
                # Check if we've reached max batches
                if max_batches and batch_count >= max_batches:
                    print(f" Reached maximum batch limit ({max_batches})")
                    break
                
                # Generate batch
                batch_result = await self.generate_batch(users_per_batch, samples_per_user)
                
                # Update and print stats
                self.update_stats(batch_result)
                self.print_stats(batch_result)
                
                batch_count += 1
                
                # Wait between batches if specified
                if batch_interval > 0 and self.running:
                    await asyncio.sleep(batch_interval)
                    
        except KeyboardInterrupt:
            print("\n=� Interrupted by user")
        except Exception as e:
            print(f"\nL Unexpected error: {e}")
        
        self.running = False
        await self.print_final_stats()
    
    async def print_final_stats(self):
        """Print final performance summary"""
        if self.stats['total_batches'] == 0:
            print("No successful batches completed.")
            return
        
        elapsed = time.time() - self.stats['start_time']
        avg_batch_time = mean(self.stats['batch_times'])
        total_samples_per_sec = self.stats['total_samples'] / elapsed
        
        print(f"\n{'='*60}")
        print(f"FINAL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total runtime:        {elapsed:.1f}s")
        print(f"Total batches:        {self.stats['total_batches']}")
        print(f"Total samples:        {self.stats['total_samples']}")
        print(f"Successful batches:   {len(self.stats['batch_times'])}")
        print(f"")
        print(f"Average batch time:   {avg_batch_time:.2f}s � {stdev(self.stats['batch_times']):.2f}s")
        print(f"Fastest batch:        {min(self.stats['batch_times']):.2f}s")
        print(f"Slowest batch:        {max(self.stats['batch_times']):.2f}s")
        print(f"")
        print(f"Overall samples/sec:  {total_samples_per_sec:.1f}")
        print(f"Overall batches/min:  {self.stats['total_batches'] * 60 / elapsed:.1f}")
        
        # Get final collector status
        await self.get_collector_status()


async def main():
    parser = argparse.ArgumentParser(description="Test collector server performance")
    parser.add_argument("--collector-url", type=str, default="http://localhost:8000",
                       help="Collector server URL")
    parser.add_argument("--users-per-batch", type=int, default=8,
                       help="Number of users per batch")
    parser.add_argument("--samples-per-user", type=int, default=4, 
                       help="Number of samples per user")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Maximum number of batches to generate (default: unlimited)")
    parser.add_argument("--batch-interval", type=float, default=0.0,
                       help="Seconds to wait between batches (default: 0)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    async with CollectorTester(args.collector_url) as tester:
        await tester.run_continuous_test(
            users_per_batch=args.users_per_batch,
            samples_per_user=args.samples_per_user,
            max_batches=args.max_batches,
            batch_interval=args.batch_interval
        )


if __name__ == "__main__":
    asyncio.run(main())