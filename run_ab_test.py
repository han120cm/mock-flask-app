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
    
    # Get CDN connection settings from environment variables
    cdn_host = os.getenv('CDN_HOST')
    cdn_user = os.getenv('CDN_USER')
    private_key_path = os.getenv('PRIVATE_KEY_PATH')
    
    # Initialize tester
    tester = ABTester(
        cache_size_mb=50,  # 50 MB cache limit
        cdn_host=cdn_host,
        cdn_user=cdn_user,
        private_key_path=private_key_path
    )
    
    # Run A/B test
    print("Starting A/B test...")
    try:
        results = tester.run_ab_test(num_iterations=5, require_real_data=False)
        
        # Print comparison report
        tester.print_comparison_report()
        
        # Export results
        tester.export_results("cdn_ab_test_results.json")
        
        print("\nTest completed successfully!")
    except Exception as e:
        if "Cannot connect to CDN" in str(e):
            print("\nError: Cannot connect to CDN. Please check connection settings.")
            print("Falling back to mock data for testing...")
            
            # Run with mock data
            results = tester.run_ab_test(num_iterations=5, require_real_data=False)
            
            # Print comparison report
            tester.print_comparison_report()
            
            # Export results
            tester.export_results("cdn_ab_test_results.json")
            
            print("\nTest completed with mock data.")
        else:
            print(f"\nError running test: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()