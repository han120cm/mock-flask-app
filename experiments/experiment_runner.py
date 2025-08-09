"""
CDN Cache Policy Experiment Runner
Runs controlled experiments to evaluate different cache eviction policies
"""

import os
import json
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages and runs cache policy experiments"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 prometheus_url: str = "http://localhost:9090",
                 output_dir: str = "experiments/results"):
        """
        Initialize experiment runner
        
        Args:
            base_url: Base URL of the CDN/application
            prometheus_url: Prometheus server URL
            output_dir: Directory to save experiment results
        """
        self.base_url = base_url
        self.prometheus_url = prometheus_url
        self.output_dir = output_dir
        self.current_experiment = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def create_experiment(self, 
                         name: str,
                         policy: str,
                         workload: str,
                         duration_minutes: int = 30,
                         warmup_minutes: int = 5,
                         cache_size_mb: int = 100,
                         seed: int = None) -> Dict[str, Any]:
        """
        Create a new experiment configuration
        """
        if seed is None:
            seed = int(time.time())
        
        experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{policy}_{workload}"
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'policy': policy,
            'workload': workload,
            'duration_minutes': duration_minutes,
            'warmup_minutes': warmup_minutes,
            'cache_size_mb': cache_size_mb,
            'seed': seed,
            'start_time': None,
            'end_time': None,
            'status': 'created',
            'metrics': {},
            'config': {
                'base_url': self.base_url,
                'prometheus_url': self.prometheus_url
            }
        }
        
        # Create experiment directory
        exp_dir = os.path.join(self.output_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        experiment['output_dir'] = exp_dir
        
        # Save initial config
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        self.current_experiment = experiment
        return experiment
    
    def run_workload(self, workload_type: str, duration_seconds: int, rps: int = 10) -> Dict[str, Any]:
        """
        Run a specific workload pattern
        
        Args:
            workload_type: Type of workload (zipf, uniform, temporal, bursty)
            duration_seconds: Duration to run the workload
            rps: Requests per second
        """
        logger.info(f"Starting workload: {workload_type} for {duration_seconds}s at {rps} RPS")
        
        workload_config = self._get_workload_config(workload_type)
        stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'response_times': [],
            'status_codes': {}
        }
        
        start_time = time.time()
        request_interval = 1.0 / rps
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            
            # Select content based on workload pattern
            content_url = self._select_content(workload_config)
            
            try:
                response = requests.get(content_url, timeout=5)
                stats['total_requests'] += 1
                
                if response.status_code == 200:
                    stats['successful_requests'] += 1
                else:
                    stats['failed_requests'] += 1
                
                # Check cache status from headers
                cache_status = response.headers.get('X-Cache-Status', 'MISS')
                if 'HIT' in cache_status:
                    stats['cache_hits'] += 1
                else:
                    stats['cache_misses'] += 1
                
                # Record response time
                response_time = response.elapsed.total_seconds()
                stats['response_times'].append(response_time)
                
                # Record status code
                status = str(response.status_code)
                stats['status_codes'][status] = stats['status_codes'].get(status, 0) + 1
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                stats['failed_requests'] += 1
                stats['total_requests'] += 1
            
            # Maintain request rate
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
        
        # Calculate statistics
        if stats['response_times']:
            stats['avg_response_time'] = np.mean(stats['response_times'])
            stats['p50_response_time'] = np.percentile(stats['response_times'], 50)
            stats['p95_response_time'] = np.percentile(stats['response_times'], 95)
            stats['p99_response_time'] = np.percentile(stats['response_times'], 99)
        
        if stats['total_requests'] > 0:
            stats['hit_ratio'] = stats['cache_hits'] / stats['total_requests']
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        
        return stats
    
    def _get_workload_config(self, workload_type: str) -> Dict[str, Any]:
        """Get configuration for different workload types"""
        
        configs = {
            'zipf': {
                'distribution': 'zipf',
                'alpha': 1.0,  # Zipf parameter (higher = more skewed)
                'content_pool_size': 1000,
                'content_types': ['image', 'video'],
                'size_distribution': 'pareto'
            },
            'uniform': {
                'distribution': 'uniform',
                'content_pool_size': 1000,
                'content_types': ['image', 'video'],
                'size_distribution': 'uniform'
            },
            'temporal': {
                'distribution': 'temporal',
                'phases': 3,  # Number of popularity phases
                'phase_duration': 300,  # Seconds per phase
                'content_pool_size': 1000,
                'hot_set_size': 50,  # Size of hot content set
                'content_types': ['image', 'video']
            },
            'bursty': {
                'distribution': 'bursty',
                'burst_probability': 0.1,
                'burst_duration': 30,  # Seconds
                'burst_content_size': 10,
                'content_pool_size': 1000,
                'content_types': ['image', 'video']
            }
        }
        
        return configs.get(workload_type, configs['uniform'])
    
    def _select_content(self, workload_config: Dict[str, Any]) -> str:
        """Select content based on workload distribution"""
        
        distribution = workload_config['distribution']
        
        if distribution == 'zipf':
            # Zipf distribution - popular items accessed more frequently
            alpha = workload_config['alpha']
            pool_size = workload_config['content_pool_size']
            
            # Generate Zipf-distributed item index
            s = np.random.zipf(alpha)
            item_index = min(s - 1, pool_size - 1)
            
        elif distribution == 'uniform':
            # Uniform distribution - all items equally likely
            pool_size = workload_config['content_pool_size']
            item_index = np.random.randint(0, pool_size)
            
        elif distribution == 'temporal':
            # Temporal locality - hot set changes over time
            current_phase = int(time.time() / workload_config['phase_duration']) % workload_config['phases']
            hot_set_size = workload_config['hot_set_size']
            
            # 80% chance to access hot set, 20% cold set
            if np.random.random() < 0.8:
                # Access from hot set for current phase
                item_index = (current_phase * hot_set_size + 
                            np.random.randint(0, hot_set_size))
            else:
                # Access from cold set
                item_index = np.random.randint(hot_set_size * workload_config['phases'],
                                              workload_config['content_pool_size'])
        
        elif distribution == 'bursty':
            # Bursty access pattern
            if np.random.random() < workload_config['burst_probability']:
                # During burst, access limited set
                item_index = np.random.randint(0, workload_config['burst_content_size'])
            else:
                # Normal access pattern
                item_index = np.random.randint(0, workload_config['content_pool_size'])
        
        else:
            # Default to uniform
            item_index = np.random.randint(0, 1000)
        
        # Select content type
        content_type = np.random.choice(workload_config.get('content_types', ['image']))
        
        # Construct URL
        if content_type == 'image':
            groups = ['trending', 'popular', 'general', 'rare']
            group = np.random.choice(groups)
            url = f"{self.base_url}/images/{group}"
        else:
            groups = ['short-clips', 'documentaries', 'tutorials', 'archived']
            group = np.random.choice(groups)
            url = f"{self.base_url}/videos/{group}"
        
        return url
    
    def collect_prometheus_metrics(self, 
                                  start_time: datetime,
                                  end_time: datetime,
                                  step: str = "30s") -> Dict[str, pd.DataFrame]:
        """
        Collect metrics from Prometheus
        """
        metrics = {}
        
        # Define queries
        queries = {
            'hit_ratio': 'rate(nginx_cache_hits_total[1m]) / (rate(nginx_cache_hits_total[1m]) + rate(nginx_cache_misses_total[1m]))',
            'byte_hit_ratio': 'rate(nginx_cache_hit_bytes_total[1m]) / (rate(nginx_cache_hit_bytes_total[1m]) + rate(nginx_cache_miss_bytes_total[1m]))',
            'request_rate': 'rate(nginx_http_requests_total[1m])',
            'error_rate': 'rate(nginx_http_requests_total{status=~"5.."}[1m])',
            'response_time_p50': 'histogram_quantile(0.5, rate(nginx_http_request_duration_seconds_bucket[1m]))',
            'response_time_p95': 'histogram_quantile(0.95, rate(nginx_http_request_duration_seconds_bucket[1m]))',
            'response_time_p99': 'histogram_quantile(0.99, rate(nginx_http_request_duration_seconds_bucket[1m]))',
            'origin_bandwidth': 'rate(nginx_upstream_bytes_received_total[1m])',
            'cache_size': 'nginx_cache_size_bytes',
            'evictions_rate': 'rate(nginx_cache_evictions_total[1m])'
        }
        
        for metric_name, query in queries.items():
            try:
                # Query Prometheus
                response = requests.get(
                    f"{self.prometheus_url}/api/v1/query_range",
                    params={
                        'query': query,
                        'start': start_time.timestamp(),
                        'end': end_time.timestamp(),
                        'step': step
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        # Convert to DataFrame
                        result = data['data']['result'][0]
                        values = result['values']
                        df = pd.DataFrame(values, columns=['timestamp', 'value'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df['value'] = pd.to_numeric(df['value'])
                        metrics[metric_name] = df
                        logger.info(f"Collected metric: {metric_name} ({len(df)} points)")
                
            except Exception as e:
                logger.error(f"Failed to collect {metric_name}: {e}")
        
        return metrics
    
    def run_experiment(self, experiment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete experiment
        """
        if experiment is None:
            experiment = self.current_experiment
        
        if experiment is None:
            raise ValueError("No experiment configured")
        
        logger.info(f"Starting experiment: {experiment['id']}")
        experiment['status'] = 'running'
        experiment['start_time'] = datetime.now().isoformat()
        
        # Total duration
        total_duration = (experiment['warmup_minutes'] + experiment['duration_minutes']) * 60
        
        # Run warmup phase
        logger.info(f"Running warmup phase for {experiment['warmup_minutes']} minutes")
        warmup_stats = self.run_workload(
            experiment['workload'],
            experiment['warmup_minutes'] * 60,
            rps=10
        )
        
        # Mark start of measurement phase
        measurement_start = datetime.now()
        
        # Run measurement phase
        logger.info(f"Running measurement phase for {experiment['duration_minutes']} minutes")
        measurement_stats = self.run_workload(
            experiment['workload'],
            experiment['duration_minutes'] * 60,
            rps=10
        )
        
        # Mark end of measurement phase
        measurement_end = datetime.now()
        
        # Collect Prometheus metrics
        logger.info("Collecting Prometheus metrics")
        prometheus_metrics = self.collect_prometheus_metrics(
            measurement_start,
            measurement_end
        )
        
        # Save results
        experiment['end_time'] = datetime.now().isoformat()
        experiment['status'] = 'completed'
        experiment['metrics'] = {
            'warmup': warmup_stats,
            'measurement': measurement_stats
        }
        
        # Save to files
        exp_dir = experiment['output_dir']
        
        # Save experiment summary
        with open(os.path.join(exp_dir, 'experiment.json'), 'w') as f:
            json.dump(experiment, f, indent=2, default=str)
        
        # Save Prometheus metrics
        for metric_name, df in prometheus_metrics.items():
            df.to_csv(os.path.join(exp_dir, f'{metric_name}.csv'), index=False)
        
        # Save measurement stats
        with open(os.path.join(exp_dir, 'measurement_stats.json'), 'w') as f:
            json.dump(measurement_stats, f, indent=2)
        
        logger.info(f"Experiment completed: {experiment['id']}")
        
        return experiment
    
    def compare_policies(self, 
                        policies: List[str],
                        workloads: List[str],
                        repetitions: int = 3,
                        **kwargs) -> pd.DataFrame:
        """
        Compare multiple policies across workloads
        """
        results = []
        
        for policy in policies:
            for workload in workloads:
                for rep in range(repetitions):
                    logger.info(f"Running {policy} with {workload} (repetition {rep+1}/{repetitions})")
                    
                    # Create experiment
                    exp = self.create_experiment(
                        name=f"{policy}_{workload}_rep{rep}",
                        policy=policy,
                        workload=workload,
                        seed=42 + rep,  # Fixed seeds for reproducibility
                        **kwargs
                    )
                    
                    # Run experiment
                    exp_result = self.run_experiment(exp)
                    
                    # Extract key metrics
                    stats = exp_result['metrics']['measurement']
                    
                    results.append({
                        'policy': policy,
                        'workload': workload,
                        'repetition': rep,
                        'hit_ratio': stats.get('hit_ratio', 0),
                        'avg_response_time': stats.get('avg_response_time', 0),
                        'p95_response_time': stats.get('p95_response_time', 0),
                        'p99_response_time': stats.get('p99_response_time', 0),
                        'success_rate': stats.get('success_rate', 0),
                        'total_requests': stats.get('total_requests', 0)
                    })
                    
                    # Small delay between experiments
                    time.sleep(5)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = df.groupby(['policy', 'workload']).agg({
            'hit_ratio': ['mean', 'std'],
            'avg_response_time': ['mean', 'std'],
            'p95_response_time': ['mean', 'std'],
            'p99_response_time': ['mean', 'std'],
            'success_rate': ['mean', 'std']
        }).round(4)
        
        # Save comparison results
        comparison_file = os.path.join(self.output_dir, 'policy_comparison.csv')
        summary.to_csv(comparison_file)
        logger.info(f"Comparison results saved to {comparison_file}")
        
        return summary


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CDN cache policy experiments")
    parser.add_argument('--policies', nargs='+', default=['lru', 'lfu', 'fifo', 'size'],
                       help='Policies to test')
    parser.add_argument('--workloads', nargs='+', default=['zipf', 'uniform', 'temporal'],
                       help='Workload patterns to test')
    parser.add_argument('--duration', type=int, default=20,
                       help='Duration of each experiment in minutes')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Warmup duration in minutes')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Cache size in MB')
    parser.add_argument('--repetitions', type=int, default=3,
                       help='Number of repetitions per configuration')
    parser.add_argument('--base-url', default='http://localhost:8000',
                       help='Base URL of the application')
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                       help='Prometheus server URL')
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(
        base_url=args.base_url,
        prometheus_url=args.prometheus_url
    )
    
    # Run comparison
    results = runner.compare_policies(
        policies=args.policies,
        workloads=args.workloads,
        repetitions=args.repetitions,
        duration_minutes=args.duration,
        warmup_minutes=args.warmup,
        cache_size_mb=args.cache_size
    )
    
    print("\nExperiment Results Summary:")
    print(results)
