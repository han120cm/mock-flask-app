#!/usr/bin/env python3
"""
Script to run A/B tests comparing cache eviction policies
"""

import sys
import os
import subprocess
import json

def run_policy_test(policy: str, region: str) -> dict:
    """Runs a single policy test and returns the metrics."""
    print(f"--- Running {policy.upper()} test for {region.upper()} ---")
    
    script_path = f"ml/{policy.lower()}-test/cdn_eviction_{policy.lower()}.py"
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return {"error": "script not found"}

    try:
        result = subprocess.run(
            ["python3", script_path, region],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Try to parse metrics from the output
        output = result.stdout
        metrics = {"policy": policy.upper(), "region": region.upper()}

        if "Eviction complete" in output or "Eviction complete" in output:
            for line in output.split('\n'):
                if "files evicted" in line or "files removed" in line:
                    metrics["files_evicted"] = int(line.split()[0])
                if "MB freed" in line:
                    metrics["mb_freed"] = float(line.split()[0])
        
        # For LFU and LRU, we can read the json log
        log_file = f"{policy.lower()}_eviction_metrics_{region.lower()}.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                # get the last entry
                if log_data:
                    metrics.update(log_data[-1])

        print(f"--- Test for {policy.upper()} completed ---")
        return metrics

    except subprocess.CalledProcessError as e:
        print(f"Error running {policy.upper()} test:")
        print(e.stderr)
        return {"error": e.stderr}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": str(e)}


def main():
    """Main function to run A/B tests"""
    print("CDN Cache Eviction Policy A/B Tester")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("Usage: python run_ab_test.py <region>")
        print("Available regions: sea, eu, us")
        sys.exit(1)
        
    region = sys.argv[1].lower()
    if region not in ['sea', 'eu', 'us']:
        print(f"Invalid region: {region}")
        sys.exit(1)

    policies = ['lru', 'lfu', 'lrb']
    all_results = []

    for policy in policies:
        results = run_policy_test(policy, region)
        all_results.append(results)

    print("\n--- A/B Test Summary ---")
    for result in all_results:
        print(json.dumps(result, indent=2))
    
    # You can add more sophisticated comparison logic here
    # For now, we just print the results.

if __name__ == "__main__":
    main()
