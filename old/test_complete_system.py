#!/usr/bin/env python3
"""
Complete System Test - OmniBio Frontend + Backend Integration
Tests the full workflow: Authentication -> Upload -> Analysis -> Results
"""

import requests
import json
import time
from pathlib import Path

# Configuration
FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"
API_KEY = "omnibio-dev-key-12345"
TEST_FILE = "ST002091_AN003415.txt"

def test_backend_health():
    """Test if backend is healthy"""
    print("🔍 Testing Backend Health...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Backend healthy: {health['status']}")
            print(f"   Database connected: {health['database_connected']}")
            return True
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def test_api_authentication():
    """Test API key authentication"""
    print("\n🔐 Testing API Authentication...")
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", headers=headers)
        if response.status_code == 200:
            print("✅ API key authentication working")
            return True
        else:
            print(f"❌ API authentication failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API authentication error: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n📤 Testing File Upload...")
    headers = {"X-API-Key": API_KEY}
    
    if not Path(TEST_FILE).exists():
        print(f"❌ Test file {TEST_FILE} not found")
        return None
    
    try:
        with open(TEST_FILE, 'rb') as f:
            files = {'file': (TEST_FILE, f, 'text/plain')}
            response = requests.post(f"{BACKEND_URL}/upload", headers=headers, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File uploaded successfully: {result['file_id']}")
            return result['file_id']
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def test_file_listing():
    """Test file listing"""
    print("\n📋 Testing File Listing...")
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BACKEND_URL}/files", headers=headers)
        if response.status_code == 200:
            files = response.json()
            print(f"✅ Listed {len(files.get('files', []))} files")
            return files
        else:
            print(f"❌ File listing failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ File listing error: {e}")
        return None

def test_analysis_creation(file_id):
    """Test analysis creation"""
    print("\n🧪 Testing Analysis Creation...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "file_ids": [file_id],
        "analysis_types": ["statistical"],
        "project_name": "Complete System Test"
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/analyze", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis created: {result['analysis_id']}")
            return result['analysis_id']
        else:
            print(f"❌ Analysis creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Analysis creation error: {e}")
        return None

def test_analysis_monitoring(analysis_id):
    """Monitor analysis progress"""
    print("\n⏱️ Monitoring Analysis Progress...")
    headers = {"X-API-Key": API_KEY}
    
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}", headers=headers)
            if response.status_code == 200:
                analysis = response.json()
                status = analysis['status']
                progress = analysis['progress']
                
                print(f"   Status: {status}, Progress: {progress:.1%}")
                
                if status == "completed":
                    print("✅ Analysis completed successfully!")
                    return True
                elif status == "failed":
                    print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
                    return False
                
                time.sleep(2)
                attempt += 1
            else:
                print(f"❌ Failed to check analysis status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Analysis monitoring error: {e}")
            return False
    
    print("❌ Analysis timeout")
    return False

def test_analyses_listing():
    """Test listing all analyses"""
    print("\n📊 Testing Analyses Listing...")
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BACKEND_URL}/analyses", headers=headers)
        if response.status_code == 200:
            analyses = response.json()
            print(f"✅ Listed {len(analyses)} analyses")
            for analysis in analyses[:3]:  # Show first 3
                print(f"   - {analysis['analysis_id'][:8]}: {analysis['status']}")
            return True
        else:
            print(f"❌ Analyses listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Analyses listing error: {e}")
        return False

def test_frontend_accessibility():
    """Test if frontend is accessible"""
    print("\n🌐 Testing Frontend Accessibility...")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend accessible")
            return True
        else:
            print(f"❌ Frontend returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend not accessible: {e}")
        return False

def main():
    """Run complete system test"""
    print("=" * 60)
    print("🚀 OMNIBIO COMPLETE SYSTEM TEST")
    print("=" * 60)
    
    # Test each component
    results = {}
    
    results['backend_health'] = test_backend_health()
    results['api_auth'] = test_api_authentication()
    results['frontend'] = test_frontend_accessibility()
    
    if results['backend_health'] and results['api_auth']:
        # Upload and analysis tests
        file_id = test_file_upload()
        test_file_listing()
        
        if file_id:
            analysis_id = test_analysis_creation(file_id)
            if analysis_id:
                results['analysis_complete'] = test_analysis_monitoring(analysis_id)
                test_analyses_listing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("Ready for production use.")
    else:
        print(f"\n⚠️ {total - passed} issues need attention.")

if __name__ == "__main__":
    main() 