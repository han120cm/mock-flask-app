#!/usr/bin/env python3
"""
Script to run A/B tests comparing cache eviction policies
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_policies.ab_tester import ABTester


def main():
    """Main function to run A/B tests"""
    print("CDN Cache Eviction Policy A/B Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = ABTester(cache_size_mb=50)  # 50 MB cache limit
    
    # Run A/B test
    print("Starting A/B test...")
    results = tester.run_ab_test(num_iterations=5)
    
    # Print comparison report
    tester.print_comparison_report()
    
    # Export results
    tester.export_results("cdn_ab_test_results.json")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()