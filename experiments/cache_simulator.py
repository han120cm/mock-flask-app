"""
Cache Simulator for Policy Evaluation
Simulates cache behavior with different eviction policies
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import OrderedDict, defaultdict
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_policies import get_policy, CachePolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheSimulator:
    """Simulates cache operations with different eviction policies"""
    
    def __init__(self, 
                 cache_size_mb: float = 100,
                 policy: str = 'lru'):
        """
        Initialize cache simulator
        
        Args:
            cache_size_mb: Maximum cache size in MB
            policy: Eviction policy name
        """
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.policy_name = policy
        self.policy = get_policy(policy, cache_size_mb=cache_size_mb)
        
        # Cache state
        self.cache = {}  # file_id -> metadata
        self.current_time = datetime.utcnow()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'bytes_evicted': 0,
            'bytes_served_from_cache': 0,
            'bytes_served_from_origin': 0,
            'requests': []
        }
    
    def reset(self):
        """Reset cache and statistics"""
        self.cache = {}
        self.current_time = datetime.utcnow()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'bytes_evicted': 0,
            'bytes_served_from_cache': 0,
            'bytes_served_from_origin': 0,
            'requests': []
        }
    
    def get(self, file_id: str, size: int = None, content_type: str = 'image') -> bool:
        """
        Simulate a cache GET request
        
        Args:
            file_id: Unique identifier for the content
            size: Size of the content in bytes
            content_type: Type of content
            
        Returns:
            True if cache hit, False if cache miss
        """
        if size is None:
            # Generate realistic size based on content type
            if content_type == 'image':
                size = np.random.lognormal(14, 1.5)  # ~1-10 MB images
            elif content_type == 'video':
                size = np.random.lognormal(17, 1.5)  # ~10-100 MB videos
            else:
                size = np.random.lognormal(12, 1.5)  # ~100KB-1MB other
            size = int(size)
        
        # Check if in cache
        if file_id in self.cache:
            # Cache hit
            self.stats['hits'] += 1
            self.stats['bytes_served_from_cache'] += size
            
            # Update access metadata
            self.cache[file_id]['last_access'] = self.current_time.isoformat()
            self.cache[file_id]['access_count'] += 1
            
            # Record request
            self.stats['requests'].append({
                'time': self.current_time.isoformat(),
                'file_id': file_id,
                'size': size,
                'hit': True,
                'evictions': 0
            })
            
            return True
        
        else:
            # Cache miss
            self.stats['misses'] += 1
            self.stats['bytes_served_from_origin'] += size
            
            # Add to cache if there's space
            self._add_to_cache(file_id, size, content_type)
            
            # Record request
            self.stats['requests'].append({
                'time': self.current_time.isoformat(),
                'file_id': file_id,
                'size': size,
                'hit': False,
                'evictions': 0  # Will be updated if eviction occurred
            })
            
            return False
    
    def _add_to_cache(self, file_id: str, size: int, content_type: str):
        """Add an item to cache, evicting if necessary"""
        
        # Check if we need to evict
        current_cache_size = sum(item['size'] for item in self.cache.values())
        
        if current_cache_size + size > self.cache_size_bytes:
            # Need to evict
            bytes_to_free = (current_cache_size + size) - self.cache_size_bytes
            self._evict(bytes_to_free)
        
        # Add to cache
        self.cache[file_id] = {
            'size': size,
            'type': content_type,
            'creation_time': self.current_time.isoformat(),
            'last_access': self.current_time.isoformat(),
            'access_count': 1
        }
    
    def _evict(self, bytes_to_free: int):
        """Evict items based on policy"""
        
        if not self.cache:
            return
        
        # Get eviction candidates from policy
        victims, metrics = self.policy.select_victims(
            self.cache,
            target_bytes=bytes_to_free,
            dry_run=False
        )
        
        # Perform evictions
        evicted_count = 0
        evicted_bytes = 0
        
        for file_id in victims:
            if file_id in self.cache:
                evicted_bytes += self.cache[file_id]['size']
                del self.cache[file_id]
                evicted_count += 1
        
        # Update statistics
        self.stats['evictions'] += evicted_count
        self.stats['bytes_evicted'] += evicted_bytes
        
        # Update last request with eviction info
        if self.stats['requests']:
            self.stats['requests'][-1]['evictions'] = evicted_count
    
    def advance_time(self, seconds: int = 1):
        """Advance simulation time"""
        self.current_time += timedelta(seconds=seconds)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return cache metrics"""
        
        total_requests = self.stats['hits'] + self.stats['misses']
        
        if total_requests == 0:
            return {
                'hit_ratio': 0,
                'byte_hit_ratio': 0,
                'miss_ratio': 0,
                'eviction_rate': 0,
                'total_requests': 0
            }
        
        total_bytes = (self.stats['bytes_served_from_cache'] + 
                      self.stats['bytes_served_from_origin'])
        
        metrics = {
            'hit_ratio': self.stats['hits'] / total_requests,
            'miss_ratio': self.stats['misses'] / total_requests,
            'byte_hit_ratio': (self.stats['bytes_served_from_cache'] / total_bytes 
                              if total_bytes > 0 else 0),
            'eviction_rate': self.stats['evictions'] / total_requests,
            'total_requests': total_requests,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_evictions': self.stats['evictions'],
            'bytes_from_cache_mb': self.stats['bytes_served_from_cache'] / (1024 * 1024),
            'bytes_from_origin_mb': self.stats['bytes_served_from_origin'] / (1024 * 1024),
            'bytes_evicted_mb': self.stats['bytes_evicted'] / (1024 * 1024),
            'cache_size_mb': sum(item['size'] for item in self.cache.values()) / (1024 * 1024),
            'cache_items': len(self.cache),
            'policy': self.policy_name
        }
        
        return metrics
    
    def simulate_trace(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate a trace of requests
        
        Args:
            trace: List of request dictionaries with 'file_id', 'size', 'type'
            
        Returns:
            Metrics dictionary
        """
        self.reset()
        
        for request in trace:
            file_id = request.get('file_id')
            size = request.get('size')
            content_type = request.get('type', 'image')
            
            self.get(file_id, size, content_type)
            
            # Optionally advance time
            if 'timestamp' in request:
                # Use provided timestamp
                self.current_time = datetime.fromisoformat(request['timestamp'])
            else:
                # Advance by small random interval
                self.advance_time(np.random.randint(1, 10))
        
        return self.get_metrics()


class WorkloadGenerator:
    """Generate synthetic workloads for testing"""
    
    @staticmethod
    def generate_zipf_workload(num_requests: int = 10000,
                              num_objects: int = 1000,
                              alpha: float = 1.0,
                              size_distribution: str = 'lognormal') -> List[Dict[str, Any]]:
        """
        Generate Zipf-distributed workload
        
        Args:
            num_requests: Number of requests to generate
            num_objects: Number of unique objects
            alpha: Zipf distribution parameter (higher = more skewed)
            size_distribution: Distribution for object sizes
        """
        trace = []
        
        # Pre-generate object sizes
        object_sizes = {}
        for i in range(num_objects):
            if size_distribution == 'lognormal':
                # Realistic web object sizes
                if i < num_objects * 0.7:  # 70% small files
                    size = int(np.random.lognormal(13, 1.5))  # ~500KB-5MB
                else:  # 30% large files
                    size = int(np.random.lognormal(16, 1.5))  # ~5-50MB
            elif size_distribution == 'uniform':
                size = np.random.randint(100_000, 10_000_000)  # 100KB-10MB
            else:
                size = 1_000_000  # Default 1MB
            
            object_sizes[f'object_{i}'] = size
        
        # Generate requests following Zipf distribution
        for _ in range(num_requests):
            # Sample from Zipf distribution
            rank = np.random.zipf(alpha)
            object_index = min(rank - 1, num_objects - 1)
            file_id = f'object_{object_index}'
            
            trace.append({
                'file_id': file_id,
                'size': object_sizes[file_id],
                'type': 'image' if object_index < num_objects * 0.8 else 'video'
            })
        
        return trace
    
    @staticmethod
    def generate_temporal_workload(num_requests: int = 10000,
                                  num_objects: int = 1000,
                                  num_phases: int = 5,
                                  hot_set_size: int = 50) -> List[Dict[str, Any]]:
        """
        Generate workload with temporal locality (changing hot sets)
        """
        trace = []
        requests_per_phase = num_requests // num_phases
        
        # Pre-generate object sizes
        object_sizes = {
            f'object_{i}': int(np.random.lognormal(14, 1.5))
            for i in range(num_objects)
        }
        
        for phase in range(num_phases):
            # Define hot set for this phase
            hot_set_start = (phase * hot_set_size) % num_objects
            hot_set = [f'object_{(hot_set_start + i) % num_objects}' 
                      for i in range(hot_set_size)]
            
            for _ in range(requests_per_phase):
                if np.random.random() < 0.8:  # 80% from hot set
                    file_id = np.random.choice(hot_set)
                else:  # 20% from cold set
                    cold_index = np.random.randint(0, num_objects)
                    file_id = f'object_{cold_index}'
                
                trace.append({
                    'file_id': file_id,
                    'size': object_sizes[file_id],
                    'type': 'image',
                    'phase': phase
                })
        
        return trace


def compare_policies_on_workload(workload: List[Dict[str, Any]],
                                policies: List[str],
                                cache_size_mb: float = 100) -> pd.DataFrame:
    """
    Compare multiple policies on the same workload
    """
    results = []
    
    for policy in policies:
        logger.info(f"Simulating {policy} policy...")
        
        # Create simulator
        sim = CacheSimulator(cache_size_mb=cache_size_mb, policy=policy)
        
        # Run simulation
        metrics = sim.simulate_trace(workload)
        
        # Store results
        results.append(metrics)
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.set_index('policy')
    
    return df


def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all policies"""
    
    # Configuration
    policies = ['lru', 'lfu', 'fifo', 'size']
    cache_sizes = [50, 100, 200]  # MB
    workload_types = ['zipf_low', 'zipf_high', 'temporal', 'uniform']
    
    all_results = []
    
    for cache_size in cache_sizes:
        logger.info(f"\n=== Cache Size: {cache_size} MB ===")
        
        for workload_type in workload_types:
            logger.info(f"\nWorkload: {workload_type}")
            
            # Generate workload
            if workload_type == 'zipf_low':
                workload = WorkloadGenerator.generate_zipf_workload(
                    num_requests=10000, alpha=0.8
                )
            elif workload_type == 'zipf_high':
                workload = WorkloadGenerator.generate_zipf_workload(
                    num_requests=10000, alpha=1.5
                )
            elif workload_type == 'temporal':
                workload = WorkloadGenerator.generate_temporal_workload(
                    num_requests=10000
                )
            else:  # uniform
                workload = WorkloadGenerator.generate_zipf_workload(
                    num_requests=10000, alpha=0.1  # Nearly uniform
                )
            
            # Compare policies
            results = compare_policies_on_workload(
                workload, policies, cache_size
            )
            
            # Add metadata
            results['cache_size_mb'] = cache_size
            results['workload'] = workload_type
            results.reset_index(inplace=True)
            
            all_results.append(results)
            
            # Print summary
            print(f"\nResults for {workload_type} with {cache_size}MB cache:")
            print(results[['policy', 'hit_ratio', 'byte_hit_ratio', 'eviction_rate']].to_string())
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    output_dir = 'experiments/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'cache_simulation_results.csv')
    final_results.to_csv(output_file, index=False)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Generate summary statistics
    summary = final_results.groupby(['policy', 'cache_size_mb', 'workload'])[
        ['hit_ratio', 'byte_hit_ratio']
    ].mean().round(4)
    
    print("\n=== SUMMARY STATISTICS ===")
    print(summary.to_string())
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache simulator for policy evaluation")
    parser.add_argument('--mode', choices=['single', 'compare', 'full'], 
                       default='compare',
                       help='Simulation mode')
    parser.add_argument('--policy', default='lru',
                       help='Policy for single mode')
    parser.add_argument('--workload', choices=['zipf', 'temporal', 'uniform'],
                       default='zipf',
                       help='Workload type')
    parser.add_argument('--requests', type=int, default=10000,
                       help='Number of requests')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Cache size in MB')
    parser.add_argument('--policies', nargs='+', 
                       default=['lru', 'lfu', 'fifo', 'size'],
                       help='Policies to compare')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single policy simulation
        logger.info(f"Running single simulation: {args.policy}")
        
        # Generate workload
        if args.workload == 'zipf':
            workload = WorkloadGenerator.generate_zipf_workload(args.requests)
        elif args.workload == 'temporal':
            workload = WorkloadGenerator.generate_temporal_workload(args.requests)
        else:
            workload = WorkloadGenerator.generate_zipf_workload(args.requests, alpha=0.1)
        
        # Run simulation
        sim = CacheSimulator(cache_size_mb=args.cache_size, policy=args.policy)
        metrics = sim.simulate_trace(workload)
        
        # Print results
        print("\nSimulation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    elif args.mode == 'compare':
        # Compare policies
        logger.info(f"Comparing policies: {args.policies}")
        
        # Generate workload
        if args.workload == 'zipf':
            workload = WorkloadGenerator.generate_zipf_workload(args.requests)
        elif args.workload == 'temporal':
            workload = WorkloadGenerator.generate_temporal_workload(args.requests)
        else:
            workload = WorkloadGenerator.generate_zipf_workload(args.requests, alpha=0.1)
        
        # Compare
        results = compare_policies_on_workload(
            workload, args.policies, args.cache_size
        )
        
        print("\nComparison Results:")
        print(results[['hit_ratio', 'byte_hit_ratio', 'eviction_rate']].to_string())
    
    else:  # full
        # Run comprehensive evaluation
        run_comprehensive_evaluation()
