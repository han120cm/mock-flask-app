#!/usr/bin/env python3
"""
Simple test script to verify routes work correctly
"""

import requests
import sys
import os

def test_route(base_url, route, expected_status=200):
    """Test a route and return the response"""
    url = f"{base_url}{route}"
    try:
        response = requests.get(url, timeout=10)
        print(f"✅ {route}: {response.status_code}")
        if response.status_code != expected_status:
            print(f"   Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
        return response.status_code == expected_status
    except Exception as e:
        print(f"❌ {route}: Error - {e}")
        return False

def main():
    # Get the base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://web-server-577176926733.us-central1.run.app"
    
    print(f"Testing routes on: {base_url}")
    print("=" * 50)
    
    # Test routes
    routes_to_test = [
        ("/", 200, "Home page"),
        ("/health", 200, "Health check"),
        ("/debug", 200, "Debug info"),
        ("/images/trending", 200, "Images trending"),
        ("/images/popular", 200, "Images popular"),
        ("/videos/short-clips", 200, "Videos short-clips"),
        ("/videos/documentaries", 200, "Videos documentaries"),
        ("/images/invalid", 404, "Invalid image group"),
        ("/videos/invalid", 404, "Invalid video group"),
    ]
    
    success_count = 0
    total_count = len(routes_to_test)
    
    for route, expected_status, description in routes_to_test:
        if test_route(base_url, route, expected_status):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_count} routes working")
    
    if success_count == total_count:
        print("🎉 All routes working correctly!")
        return 0
    else:
        print("⚠️  Some routes have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 