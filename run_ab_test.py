#!/usr/bin/env python3
"""
Script to run A/B tests comparing cache eviction policies.
This script requires a live CDN connection and does not use mock data.
"""

import sys
import os
import subprocess
import json

def run_command(command: list, description: str) -> bool:
    """Runs a command and handles errors."""
    print(f"--- Running: {description} ---")
    try:
        # Explicitly use bash to execute the command
        result = subprocess.run(["bash", "-c", " ".join(command)], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"--- {description} completed successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {description} ---")
        print(e.stderr)
        return False

def run_policy_test(policy: str, region: str) -> dict:
    """Runs a single policy test and returns the metrics."""
    print(f"\n{'='*20} Starting test for {policy.upper()} policy in {region.upper()} region {'='*20}")

    # 1. Delete cache
    if not run_command(["bash", "ml/simulation/cdn-delete.sh", region], "Delete cache"):
        return {"policy": policy, "region": region, "status": "failed", "step": "delete"}

    # 2. Simulate traffic
    if not run_command(["bash", "ml/simulation/cdn_trend_simulation_zipf.sh"], "Simulate traffic (pre-eviction)нка"):
        return {"policy": policy, "region": region, "status": "failed", "step": "pre-eviction simulation"}

    # 3. Evict
    eviction_script = f"ml/{policy}-test/cdn_eviction_{policy}.py"
    if not run_command(["python3", eviction_script, region], f"Evict using {policy.upper()}"):
        return {"policy": policy, "region": region, "status": "failed", "step": "eviction"}

    # 4. Simulate traffic again
    if not run_command(["bash", "ml/simulation/cdn_trend_simulation_zipf.sh"], "Simulate traffic (post-eviction)"):
        return {"policy": policy, "region": region, "status": "failed", "step": "post-eviction simulation"}

    # 5. Collect metrics
    # As the scripts are not designed to return structured metrics, we will check for the log files
    log_file = f"{policy.lower()}_eviction_metrics_{region.lower()}.json"
    metrics = {"policy": policy, "region": region, "status": "success"}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            if log_data:
                metrics.update(log_data[-1])
    
    print(f"--- Test for {policy.upper()} completed successfully ---")
    return metrics


def main():
    """Main function to run A/B tests"""
    print("CDN Cache Eviction Policy A/B Tester")
    print("This script runs tests against a live CDN following a specific flow:")
    print("1. Delete cache -> 2. Simulate traffic -> 3. Evict -> 4. Simulate traffic")
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
        if results["status"] == "failed":
            print(f"--- Test for {policy.upper()} failed at step: {results['step']}. Halting tests. ---")
            break

    print("\n--- A/B Test Execution Summary ---")
    for result in all_results:
        print(json.dumps(result, indent=2))
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()