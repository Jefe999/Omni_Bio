#!/usr/bin/env python3
"""
Test script for scaling integration
Tests that the new user-controlled scaling options work correctly
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
API_KEY = "omnibio-dev-key-12345"

def test_health():
    """Test API health"""
    response = requests.get(f"{API_BASE}/health")
    print(f"✅ API Health: {response.status_code}")
    return response.status_code == 200

def test_upload_file():
    """Upload a test file"""
    # Find the ST002091 file
    test_file = None
    possible_paths = [
        "ST002091_AN003415.txt",
        "biomarker/data/ST002091_AN003415.txt",
        "../biomarker/data/ST002091_AN003415.txt"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            test_file = path
            break
    
    if not test_file:
        print("❌ Test file not found!")
        return None
    
    print(f"📁 Uploading: {test_file}")
    
    headers = {"X-API-Key": API_KEY}
    
    with open(test_file, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_BASE}/upload", files=files, headers=headers)
    
    if response.status_code == 200:
        file_info = response.json()
        print(f"✅ File uploaded: {file_info['file_id']}")
        return file_info['file_id']
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return None

def test_scaling_analysis(file_id, scaling_method="pareto", log_transform=False):
    """Test analysis with specific scaling parameters"""
    print(f"\n🔬 Testing scaling: {scaling_method}, log_transform: {log_transform}")
    
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    analysis_request = {
        "file_ids": [file_id],
        "analysis_types": ["statistical"],
        "project_name": f"Scaling Test - {scaling_method}",
        "scaling_method": scaling_method,
        "log_transform": log_transform,
        "log_base": "log10",
        "p_value_threshold": 0.05
    }
    
    response = requests.post(f"{API_BASE}/analyze", json=analysis_request, headers=headers)
    
    if response.status_code == 200:
        analysis_info = response.json()
        analysis_id = analysis_info['analysis_id']
        print(f"✅ Analysis started: {analysis_id}")
        
        # Wait for completion
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            status_response = requests.get(f"{API_BASE}/analyses/{analysis_id}", headers=headers)
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   Progress: {status['progress']:.1%} - {status['message']}")
                
                if status['status'] == 'completed':
                    print(f"✅ Analysis completed!")
                    return analysis_id, status
                elif status['status'] == 'failed':
                    print(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
                    return analysis_id, status
        
        print(f"⏰ Analysis timed out")
        return analysis_id, None
    
    else:
        print(f"❌ Analysis failed to start: {response.status_code} - {response.text}")
        return None, None

def test_different_scaling_methods(file_id):
    """Test different scaling methods"""
    
    scaling_methods = [
        ("none", False),
        ("pareto", False),
        ("standard", False),
        ("pareto", True),  # With log transform
        ("robust", False),
        ("minmax", False)
    ]
    
    results = {}
    
    for scaling_method, log_transform in scaling_methods:
        analysis_id, status = test_scaling_analysis(file_id, scaling_method, log_transform)
        
        if status and status['status'] == 'completed':
            # Check if scaling info was saved
            scaling_key = f"{scaling_method}_log{log_transform}"
            results[scaling_key] = {
                'analysis_id': analysis_id,
                'status': status,
                'scaling_method': scaling_method,
                'log_transform': log_transform
            }
        
        time.sleep(2)  # Brief pause between tests
    
    return results

def main():
    """Run scaling integration tests"""
    print("🧪 Testing Scaling Integration")
    print("=" * 50)
    
    # Test 1: API Health
    if not test_health():
        print("❌ API not healthy, exiting")
        return
    
    # Test 2: Upload file
    file_id = test_upload_file()
    if not file_id:
        print("❌ File upload failed, exiting")
        return
    
    # Test 3: Test different scaling methods
    print("\n🔬 Testing Different Scaling Methods")
    results = test_different_scaling_methods(file_id)
    
    # Test 4: Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    if results:
        print(f"✅ Successful tests: {len(results)}")
        for key, result in results.items():
            method = result['scaling_method']
            log_tf = result['log_transform']
            print(f"   • {method} (log: {log_tf}): {result['analysis_id'][:8]}...")
            
        # Save detailed results
        with open("scaling_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved: scaling_test_results.json")
        print("\n🎉 Scaling integration test completed successfully!")
        
    else:
        print("❌ No successful tests")

if __name__ == "__main__":
    main() 