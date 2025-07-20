#!/usr/bin/env python3
"""
Deployment verification script for Cloud Run
"""

import requests
import json
import sys
import time

def check_deployment(base_url):
    """Check if the deployment is working correctly"""
    print(f"🔍 Checking deployment at: {base_url}")
    print("=" * 60)
    
    # Test basic endpoints
    endpoints = [
        ("/", "Home page"),
        ("/health", "Health check"),
        ("/debug", "Debug info"),
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=15)
            print(f"✅ {description}: {response.status_code}")
            if endpoint == "/debug":
                try:
                    debug_info = response.json()
                    print(f"   Database: {debug_info.get('database_path', 'N/A')}")
                    print(f"   Environment: {debug_info.get('environment', 'N/A')}")
                    print(f"   GCS Available: {debug_info.get('gcs_available', 'N/A')}")
                    print(f"   CDN Available: {debug_info.get('cdn_available', 'N/A')}")
                except:
                    print(f"   Response: {response.text[:100]}...")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Testing image routes...")
    
    # Test image routes
    image_groups = ["trending", "popular", "general", "rare"]
    for group in image_groups:
        try:
            response = requests.get(f"{base_url}/images/{group}", timeout=15)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} /images/{group}: {response.status_code}")
        except Exception as e:
            print(f"❌ /images/{group}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Testing video routes...")
    
    # Test video routes
    video_groups = ["short-clips", "documentaries", "tutorials", "archived"]
    for group in video_groups:
        try:
            response = requests.get(f"{base_url}/videos/{group}", timeout=15)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} /videos/{group}: {response.status_code}")
        except Exception as e:
            print(f"❌ /videos/{group}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Testing error handling...")
    
    # Test error handling
    error_tests = [
        ("/images/invalid", 404, "Invalid image group"),
        ("/videos/invalid", 404, "Invalid video group"),
        ("/nonexistent", 404, "Non-existent route"),
    ]
    
    for endpoint, expected_status, description in error_tests:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=15)
            status = "✅" if response.status_code == expected_status else "❌"
            print(f"{status} {description}: {response.status_code} (expected {expected_status})")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_deployment.py <base_url>")
        print("Example: python verify_deployment.py https://web-server-577176926733.us-central1.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    check_deployment(base_url)

if __name__ == "__main__":
    main() 