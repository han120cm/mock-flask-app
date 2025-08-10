"""A/B Testing Framework for Cache Eviction Policies"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from .lru import LRUPolicy
from .lfu import LFUPolicy
from .lrb import LRBPolicy


class ABTester:
    """A/B Testing framework for comparing cache eviction policies"""
    
    def __init__(self, cache_size_mb: float = 100):
        """
        Initialize A/B tester
        
        Args:
            cache_size_mb: Cache size limit in MB
        """
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.policies = {
            'LRU': LRUPolicy(cache_size_mb),
            'LFU': LFUPolicy(cache_size_mb),
            'LRB': LRBPolicy(cache_size_mb)
        }
        self.test_results = []
    
    def generate_mock_cache_index(self, num_items: int = 100) -> Dict[str, Any]:
        """
        Generate a realistic mock cache index for testing
        
        Args:
            num_items: Number of cache items to generate
            
        Returns:
            Dictionary representing cache index
        """
        cache_index = {}
        base_time = datetime.utcnow()
        
        for i in range(num_items):
            # Generate varied access patterns
            age_hours = random.randint(1, 720)  # Up to 30 days old
            # Skewed access count distribution (Zipf-like)
            access_count = random.choice([1] * 50 + [2] * 25 + [3] * 15 + [5] * 5 + [10] * 3 + [20, 50])
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
    
    def calculate_hit_ratio(self, cache_index: Dict[str, Any], 
                          evicted_files: List[str]) -> float:
        """
        Calculate cache hit ratio based on evicted files
        
        Args:
            cache_index: Original cache index
            evicted_files: List of evicted file IDs
            
        Returns:
            Hit ratio (0.0 to 1.0)
        """
        if not cache_index:
            return 0.0
            
        evicted_set = set(evicted_files)
        total_items = len(cache_index)
        remaining_items = total_items - len(evicted_set)
        
        return remaining_items / total_items if total_items > 0 else 0.0
    
    def calculate_byte_hit_ratio(self, cache_index: Dict[str, Any], 
                               evicted_files: List[str]) -> float:
        """
        Calculate byte hit ratio (weighted by file size)
        
        Args:
            cache_index: Original cache index
            evicted_files: List of evicted file IDs
            
        Returns:
            Byte hit ratio (0.0 to 1.0)
        """
        if not cache_index:
            return 0.0
            
        evicted_set = set(evicted_files)
        total_size = sum(item['size'] for item in cache_index.values())
        
        if total_size == 0:
            return 0.0
            
        evicted_size = sum(
            cache_index[fid]['size'] for fid in evicted_files 
            if fid in cache_index
        )
        remaining_size = total_size - evicted_size
        
        return remaining_size / total_size
    
    def run_single_test(self, cache_index: Dict[str, Any], 
                       policy_name: str) -> Dict[str, Any]:
        """
        Run a single test with one policy
        
        Args:
            cache_index: Cache index to test
            policy_name: Name of policy to test ('LRU', 'LFU', 'LRB')
            
        Returns:
            Test results dictionary
        """
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        policy = self.policies[policy_name]
        
        # Run eviction
        victims, metrics = policy.select_victims(cache_index, dry_run=True)
        
        # Calculate metrics
        hit_ratio = self.calculate_hit_ratio(cache_index, victims)
        byte_hit_ratio = self.calculate_byte_hit_ratio(cache_index, victims)
        
        # Enhanced metrics
        total_size = sum(item['size'] for item in cache_index.values())
        evicted_size = metrics['evicted_bytes']
        remaining_size = total_size - evicted_size
        
        result = {
            'policy': policy_name,
            'hit_ratio': hit_ratio,
            'byte_hit_ratio': byte_hit_ratio,
            'items_evicted': len(victims),
            'bytes_evicted_mb': evicted_size / (1024 * 1024),
            'items_remaining': len(cache_index) - len(victims),
            'bytes_remaining_mb': remaining_size / (1024 * 1024),
            'total_items': len(cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return result
    
    def run_ab_test(self, cache_index: Dict[str, Any] = None, 
                   num_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Run A/B test comparing all policies
        
        Args:
            cache_index: Cache index to test (generates mock if None)
            num_iterations: Number of test iterations
            
        Returns:
            List of test results
        """
        if cache_index is None:
            cache_index = self.generate_mock_cache_index(100)
        
        all_results = []
        
        print(f"Running A/B test with {num_iterations} iterations...")
        print(f"Cache size: {self.cache_size_mb} MB")
        print(f"Policies: {', '.join(self.policies.keys())}")
        print("-" * 60)
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Test each policy
            for policy_name in self.policies.keys():
                try:
                    result = self.run_single_test(cache_index, policy_name)
                    result['iteration'] = iteration + 1
                    all_results.append(result)
                    print(f"  {policy_name}: {result['hit_ratio']:.3f} hit ratio")
                except Exception as e:
                    print(f"  {policy_name}: ERROR - {e}")
                    # Add error result
                    all_results.append({
                        'policy': policy_name,
                        'iteration': iteration + 1,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        self.test_results = all_results
        return all_results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all test results
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.test_results:
            return pd.DataFrame()
        
        # Filter out error results
        valid_results = [r for r in self.test_results if 'error' not in r]
        
        if not valid_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(valid_results)
        
        # Group by policy and calculate statistics
        summary = df.groupby('policy').agg({
            'hit_ratio': ['mean', 'std', 'min', 'max'],
            'byte_hit_ratio': ['mean', 'std', 'min', 'max'],
            'items_evicted': ['mean', 'std'],
            'bytes_evicted_mb': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        return summary
    
    def print_comparison_report(self):
        """Print a detailed comparison report"""
        if not self.test_results:
            print("No test results available. Run ab_test() first.")
            return
        
        print("\n" + "=" * 80)
        print("CACHE EVICTION POLICY A/B TEST RESULTS")
        print("=" * 80)
        
        # Summary statistics
        summary = self.get_summary_statistics()
        if not summary.empty:
            print("\nSUMMARY STATISTICS (across all iterations):")
            print("-" * 60)
            print(summary)
        
        # Detailed comparison
        print("\nDETAILED COMPARISON:")
        print("-" * 60)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([r for r in self.test_results if 'error' not in r])
        
        # Best performers for each metric
        if not df.empty:
            best_hit = df.loc[df['hit_ratio'].idxmax()]
            best_byte = df.loc[df['byte_hit_ratio'].idxmax()]
            
            print(f"Highest Hit Ratio: {best_hit['policy']} ({best_hit['hit_ratio']:.3f})")
            print(f"Highest Byte Hit Ratio: {best_byte['policy']} ({best_byte['byte_hit_ratio']:.3f})")
            
            # Average performance by policy
            print("\nAVERAGE PERFORMANCE BY POLICY:")
            policy_avg = df.groupby('policy')[['hit_ratio', 'byte_hit_ratio']].mean().round(4)
            print(policy_avg)
        
        print("\n" + "=" * 80)
        print("INTERPRETATION GUIDE:")
        print("-" * 40)
        print("• Hit Ratio: Proportion of items that remain in cache")
        print("• Byte Hit Ratio: Proportion of total data that remains in cache")
        print("• Higher values indicate better cache performance")
        print("• LRB should outperform LRU/LFU in most real-world scenarios")
        print("=" * 80)
    
    def export_results(self, filename: str = "ab_test_results.json"):
        """
        Export test results to JSON file
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"Results exported to {filename}")