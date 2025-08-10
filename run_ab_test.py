#!/usr/bin/env python3
"""
Script to run A/B tests comparing cache eviction policies.
This script requires a live CDN connection and does not use mock data.
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
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    try:
        result = subprocess.run(
            ["python3", script_path, region],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        metrics = {"policy": policy.upper(), "region": region.upper(), "status": "success"}
        print(output) # Print the full output of the script

        # You can add more sophisticated parsing here if needed
        # For now, we just capture the success status.

        print(f"--- Test for {policy.upper()} completed successfully ---")
        return metrics

    except subprocess.CalledProcessError as e:
        print(f"--- Error running {policy.upper()} test for {region.upper()} ---")
        print("Error: The script failed to execute. This might be due to a problem with the CDN connection or the script itself.")
        print("\n--- stderr from script ---")
        print(e.stderr)
        print("--------------------------")
        sys.exit(1)


def main():
    """Main function to run A/B tests"""
    print("CDN Cache Eviction Policy A/B Tester")
    print("This script runs tests against a live CDN and requires a working connection.")
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

    print("\n--- A/B Test Execution Summary ---")
    for result in all_results:
        print(json.dumps(result, indent=2))
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()