#!/usr/bin/env python3
"""
Simple test script to demonstrate cache eviction policies
No external dependencies required except standard library
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Mock cache index for testing
def generate_mock_cache_index(num_items: int = 100) -> Dict[str, Any]:
    """Generate a mock cache index for testing"""
    cache_index = {}
    base_time = datetime.utcnow()
    
    for i in range(num_items):
        # Generate varied access patterns
        age_hours = random.randint(1, 720)  # Up to 30 days old
        access_count = random.choice([1, 1, 2, 3, 5, 10, 20, 50])  # Skewed access
        size = random.choice([
            100_000,    # 100KB
            500_000,    # 500KB
            1_000_000,  # 1MB
            5_000_000,  # 5MB
            10_000_000, # 10MB
            50_000_000  # 50MB
        ])
        
        file_id = f"file_{i:04d}"
        last_access = base_time - timedelta(hours=age_hours)
        creation = last_access - timedelta(hours=random.randint(0, 100))
        
        cache_index[file_id] = {
            'size': size,
            'last_access': last_access.isoformat(),
            'creation_time': creation.isoformat(),
            'access_count': access_count,
            'type': random.choice(['image', 'video', 'other'])
        }
    
    return cache_index

def calculate_cache_metrics(cache_index: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic metrics for the cache"""
    total_size = sum(item['size'] for item in cache_index.values())
    total_items = len(cache_index)
    avg_size = total_size / total_items if total_items > 0 else 0
    
    access_counts = [item['access_count'] for item in cache_index.values()]
    avg_access = sum(access_counts) / len(access_counts) if access_counts else 0
    
    return {
        'total_items': total_items,
        'total_size_mb': total_size / (1024 * 1024),
        'avg_size_mb': avg_size / (1024 * 1024),
        'avg_access_count': avg_access,
        'min_access_count': min(access_counts) if access_counts else 0,
        'max_access_count': max(access_counts) if access_counts else 0
    }

def score_lru(cache_index: Dict[str, Any]) -> List[tuple]:
    """Score items using LRU policy (older = evict first)"""
    scored_items = []
    now = datetime.utcnow()
    
    for file_id, metadata in cache_index.items():
        last_access = datetime.fromisoformat(metadata['last_access'])
        age_seconds = (now - last_access).total_seconds()
        scored_items.append((file_id, age_seconds, metadata['size']))
    
    # Sort by score (higher = evict first)
    return sorted(scored_items, key=lambda x: x[1], reverse=True)

def score_lfu(cache_index: Dict[str, Any]) -> List[tuple]:
    """Score items using LFU policy (less frequent = evict first)"""
    scored_items = []
    
    for file_id, metadata in cache_index.items():
        access_count = metadata.get('access_count', 0)
        # Lower access count = higher eviction score
        score = 1.0 / (access_count + 1)
        scored_items.append((file_id, score, metadata['size']))
    
    return sorted(scored_items, key=lambda x: x[1], reverse=True)

def score_fifo(cache_index: Dict[str, Any]) -> List[tuple]:
    """Score items using FIFO policy (oldest creation = evict first)"""
    scored_items = []
    now = datetime.utcnow()
    
    for file_id, metadata in cache_index.items():
        creation = datetime.fromisoformat(metadata.get('creation_time', metadata['last_access']))
        age_seconds = (now - creation).total_seconds()
        scored_items.append((file_id, age_seconds, metadata['size']))
    
    return sorted(scored_items, key=lambda x: x[1], reverse=True)

def score_size_aware(cache_index: Dict[str, Any]) -> List[tuple]:
    """Score items using size-aware policy (large + infrequent = evict first)"""
    scored_items = []
    now = datetime.utcnow()
    
    for file_id, metadata in cache_index.items():
        size = metadata['size']
        access_count = metadata.get('access_count', 0)
        last_access = datetime.fromisoformat(metadata['last_access'])
        age_hours = (now - last_access).total_seconds() / 3600
        
        # Value per byte: higher access + recent = higher value
        recency_factor = 1.0 / (1.0 + age_hours / 24)  # Decay over days
        value_per_byte = (access_count + 1) * recency_factor / size
        
        # Eviction score: inverse of value (lower value = evict first)
        score = 1.0 / (value_per_byte + 0.0001)
        scored_items.append((file_id, score, size))
    
    return sorted(scored_items, key=lambda x: x[1], reverse=True)

def simulate_eviction(cache_index: Dict[str, Any], 
                      policy: str, 
                      target_size_mb: float = 100) -> Dict[str, Any]:
    """Simulate cache eviction with given policy"""
    
    # Get scoring function
    scoring_functions = {
        'lru': score_lru,
        'lfu': score_lfu,
        'fifo': score_fifo,
        'size': score_size_aware
    }
    
    if policy not in scoring_functions:
        raise ValueError(f"Unknown policy: {policy}")
    
    # Score all items
    scored_items = scoring_functions[policy](cache_index)
    
    # Calculate current size
    current_size = sum(item['size'] for item in cache_index.values())
    target_size = target_size_mb * 1024 * 1024
    
    # Select items to evict
    evicted = []
    evicted_size = 0
    remaining_items = []
    
    for file_id, score, size in scored_items:
        if current_size - evicted_size <= target_size:
            remaining_items.append(file_id)
        else:
            evicted.append(file_id)
            evicted_size += size
    
    return {
        'policy': policy,
        'evicted_count': len(evicted),
        'evicted_size_mb': evicted_size / (1024 * 1024),
        'remaining_count': len(remaining_items),
        'remaining_size_mb': (current_size - evicted_size) / (1024 * 1024),
        'evicted_items': evicted[:10]  # Show first 10 evicted
    }

def main():
    """Main test function"""
    print("=" * 80)
    print("CDN CACHE EVICTION POLICY TEST")
    print("=" * 80)
    print()
    
    # Generate mock cache
    print("Generating mock cache index with 100 items...")
    cache_index = generate_mock_cache_index(100)
    
    # Calculate metrics
    metrics = calculate_cache_metrics(cache_index)
    print("\nCache Metrics:")
    print(f"  Total items: {metrics['total_items']}")
    print(f"  Total size: {metrics['total_size_mb']:.2f} MB")
    print(f"  Average size: {metrics['avg_size_mb']:.2f} MB")
    print(f"  Average access count: {metrics['avg_access_count']:.1f}")
    print(f"  Access count range: {metrics['min_access_count']} - {metrics['max_access_count']}")
    print()
    
    # Test each policy
    policies = ['lru', 'lfu', 'fifo', 'size']
    target_cache_size = 50  # MB
    
    print(f"Testing eviction policies (target cache size: {target_cache_size} MB)")
    print("-" * 60)
    
    results = []
    for policy in policies:
        result = simulate_eviction(cache_index, policy, target_cache_size)
        results.append(result)
        
        print(f"\n{policy.upper()} Policy:")
        print(f"  Evicted: {result['evicted_count']} items ({result['evicted_size_mb']:.2f} MB)")
        print(f"  Remaining: {result['remaining_count']} items ({result['remaining_size_mb']:.2f} MB)")
        print(f"  First evicted: {', '.join(result['evicted_items'][:5])}")
    
    # Compare policies
    print("\n" + "=" * 80)
    print("POLICY COMPARISON SUMMARY")
    print("-" * 60)
    
    # Find which policy evicts fewest items
    min_evicted = min(results, key=lambda x: x['evicted_count'])
    print(f"Fewest items evicted: {min_evicted['policy'].upper()} ({min_evicted['evicted_count']} items)")
    
    # Find which policy evicts least data
    min_size = min(results, key=lambda x: x['evicted_size_mb'])
    print(f"Least data evicted: {min_size['policy'].upper()} ({min_size['evicted_size_mb']:.2f} MB)")
    
    # Show detailed comparison
    print("\nDetailed Comparison:")
    print(f"{'Policy':<10} {'Items Evicted':<15} {'Data Evicted (MB)':<20}")
    print("-" * 45)
    for result in results:
        print(f"{result['policy'].upper():<10} {result['evicted_count']:<15} {result['evicted_size_mb']:<20.2f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("\nThis demonstrates how different cache eviction policies behave:")
    print("- LRU: Evicts least recently used items")
    print("- LFU: Evicts least frequently used items")
    print("- FIFO: Evicts oldest items (by creation time)")
    print("- Size-aware: Considers both size and access patterns")
    print("\nFor your thesis, you should:")
    print("1. Run these policies on real CDN traces")
    print("2. Measure hit ratio, byte hit ratio, and latency")
    print("3. Compare with your ML-based LRB policy")
    print("4. Show statistical significance of improvements")

if __name__ == "__main__":
    main()
