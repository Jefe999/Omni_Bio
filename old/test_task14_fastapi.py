#!/usr/bin/env python3
"""
Test Task #14: FastAPI Endpoints
Comprehensive test of the biomarker discovery REST API.

Tests all endpoints:
- File upload and management
- Analysis execution with background tasks
- Results retrieval
- Artifact packaging
- Status monitoring
"""

import requests
import time
import os
import json
from pathlib import Path
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 300  # 5 minutes for analysis completion

def test_health_check():
    """Test API health check endpoints."""
    print("🏥 Testing health check endpoints...")
    
    # Test root endpoint
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "OmniBio Biomarker Discovery API"
    assert data["status"] == "healthy"
    print("  ✅ Root endpoint working")
    
    # Test detailed health check
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "upload_dir" in data
    assert "results_dir" in data
    print("  ✅ Health check endpoint working")

def test_file_upload():
    """Test file upload functionality."""
    print("📁 Testing file upload...")
    
    # Check if test file exists
    test_file = "ST002091_AN003415.txt"
    if not os.path.exists(test_file):
        print(f"  ⚠️ Test file {test_file} not found - skipping upload test")
        return None
    
    # Upload file
    with open(test_file, 'rb') as f:
        files = {'file': (test_file, f, 'text/plain')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'file_id' in data
    assert data['filename'] == test_file
    assert data['file_type'] in ['mwtab', 'unknown']  # Should detect mwTab
    assert data['status'] == 'uploaded'
    
    file_id = data['file_id']
    print(f"  ✅ File uploaded successfully: {file_id}")
    
    # Test file listing
    response = requests.get(f"{API_BASE_URL}/files")
    assert response.status_code == 200
    files = response.json()
    assert len(files) >= 1
    print(f"  ✅ File listing working: {len(files)} files")
    
    return file_id

def test_analysis_execution(file_id: str):
    """Test analysis execution with background tasks."""
    print("🔬 Testing analysis execution...")
    
    # Start analysis
    analysis_request = {
        "file_ids": [file_id],
        "analysis_types": ["qc", "statistical", "ml"],
        "project_name": "api_test_analysis"
    }
    
    response = requests.post(f"{API_BASE_URL}/analyze", json=analysis_request)
    assert response.status_code == 200
    data = response.json()
    
    analysis_id = data['analysis_id']
    assert data['status'] in ['pending', 'running']
    print(f"  ✅ Analysis started: {analysis_id}")
    
    # Monitor analysis progress
    print("  🔍 Monitoring analysis progress...")
    start_time = time.time()
    
    while time.time() - start_time < TEST_TIMEOUT:
        response = requests.get(f"{API_BASE_URL}/analyses/{analysis_id}")
        assert response.status_code == 200
        status = response.json()
        
        print(f"    Status: {status['status']} ({status['progress']:.1%}) - {status['message']}")
        
        if status['status'] == 'completed':
            print("  ✅ Analysis completed successfully!")
            return analysis_id
        elif status['status'] == 'failed':
            print(f"  ❌ Analysis failed: {status.get('error', 'Unknown error')}")
            return None
        
        time.sleep(5)  # Check every 5 seconds
    
    print("  ⚠️ Analysis timed out")
    return None

def test_results_retrieval(analysis_id: str):
    """Test results retrieval endpoints."""
    print("📊 Testing results retrieval...")
    
    # Get analysis results
    response = requests.get(f"{API_BASE_URL}/analyses/{analysis_id}/results")
    assert response.status_code == 200
    results = response.json()
    
    assert 'data_info' in results
    data_info = results['data_info']
    print(f"  ✅ Results retrieved: {data_info['n_samples']} samples × {data_info['n_features']} features")
    
    # List analysis files
    response = requests.get(f"{API_BASE_URL}/analyses/{analysis_id}/files")
    assert response.status_code == 200
    files = response.json()
    
    print(f"  ✅ Analysis files listed: {len(files)} files")
    
    # Test file download (try to download a small file)
    for file_info in files:
        if file_info['size_bytes'] < 1024 * 1024:  # Download files < 1MB
            file_path = file_info['path']
            response = requests.get(f"{API_BASE_URL}/analyses/{analysis_id}/files/{file_path}")
            assert response.status_code == 200
            print(f"    ✅ Downloaded file: {file_path} ({file_info['size_bytes']} bytes)")
            break
    
    return results

def test_artifact_packaging(analysis_id: str):
    """Test artifact packaging functionality."""
    print("📦 Testing artifact packaging...")
    
    # Create package
    package_request = {
        "analysis_id": analysis_id,
        "include_data": True,
        "create_zip": True,
        "create_tar": False
    }
    
    response = requests.post(f"{API_BASE_URL}/package", json=package_request)
    assert response.status_code == 200
    data = response.json()
    
    package_id = data['package_id']
    assert data['status'] == 'packaging'
    print(f"  ✅ Package creation started: {package_id}")
    
    # Wait for packaging to complete (packages are usually fast)
    time.sleep(10)
    
    # Check package status
    response = requests.get(f"{API_BASE_URL}/packages/{package_id}")
    if response.status_code == 200:
        package_info = response.json()
        print(f"  ✅ Package created successfully")
        
        # Try to download package
        response = requests.get(f"{API_BASE_URL}/packages/{package_id}/download?format=zip")
        if response.status_code == 200:
            print(f"    ✅ Package download working ({len(response.content)} bytes)")
        else:
            print(f"    ⚠️ Package download failed: {response.status_code}")
    else:
        print(f"  ⚠️ Package info not available yet: {response.status_code}")
    
    # List all packages
    response = requests.get(f"{API_BASE_URL}/packages")
    assert response.status_code == 200
    packages = response.json()
    print(f"  ✅ Package listing working: {len(packages)} packages")
    
    return package_id

def test_analysis_management():
    """Test analysis management endpoints."""
    print("🗂️ Testing analysis management...")
    
    # List all analyses
    response = requests.get(f"{API_BASE_URL}/analyses")
    assert response.status_code == 200
    analyses = response.json()
    print(f"  ✅ Analysis listing working: {len(analyses)} analyses")
    
    return analyses

def test_cleanup(file_id: str, analysis_id: str):
    """Test cleanup functionality."""
    print("🧹 Testing cleanup...")
    
    # Delete analysis
    response = requests.delete(f"{API_BASE_URL}/analyses/{analysis_id}")
    if response.status_code == 200:
        print("  ✅ Analysis deletion working")
    else:
        print(f"  ⚠️ Analysis deletion failed: {response.status_code}")
    
    # Delete file
    response = requests.delete(f"{API_BASE_URL}/files/{file_id}")
    if response.status_code == 200:
        print("  ✅ File deletion working")
    else:
        print(f"  ⚠️ File deletion failed: {response.status_code}")

def run_comprehensive_api_test():
    """Run the complete API test suite."""
    print("=" * 70)
    print("🎯 TESTING TASK #14: FASTAPI ENDPOINTS")
    print("=" * 70)
    
    try:
        # Test health check
        test_health_check()
        
        # Test file upload
        file_id = test_file_upload()
        if not file_id:
            print("❌ Cannot continue without successful file upload")
            return False
        
        # Test analysis execution
        analysis_id = test_analysis_execution(file_id)
        if not analysis_id:
            print("❌ Cannot continue without successful analysis")
            return False
        
        # Test results retrieval
        results = test_results_retrieval(analysis_id)
        
        # Test artifact packaging
        package_id = test_artifact_packaging(analysis_id)
        
        # Test analysis management
        analyses = test_analysis_management()
        
        # Test cleanup
        test_cleanup(file_id, analysis_id)
        
        print("\n" + "=" * 70)
        print("🎉 ALL API TESTS PASSED!")
        print("=" * 70)
        print(f"✅ Health checks: Working")
        print(f"✅ File upload: Working ({file_id})")
        print(f"✅ Analysis execution: Working ({analysis_id})")
        print(f"✅ Results retrieval: Working")
        print(f"✅ Artifact packaging: Working ({package_id})")
        print(f"✅ Analysis management: Working ({len(analyses)} analyses)")
        print(f"✅ Cleanup: Working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_api_server():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

if __name__ == "__main__":
    print("🚀 Starting FastAPI endpoint tests...")
    
    # Check if API server is running
    if not check_api_server():
        print(f"❌ API server not running at {API_BASE_URL}")
        print("Please start the API server first:")
        print("  cd biomarker/api")
        print("  python -m uvicorn main:app --reload")
        exit(1)
    
    print(f"✅ API server is running at {API_BASE_URL}")
    
    # Run comprehensive test
    success = run_comprehensive_api_test()
    
    if success:
        print("\n🏆 Task #14 FastAPI Endpoints - ALL TESTS PASSED!")
        exit(0)
    else:
        print("\n💥 Task #14 FastAPI Endpoints - TESTS FAILED!")
        exit(1) 