#!/usr/bin/env python3
"""
Live OmniBio Demo - Real-time testing and visualization
Run this to see your system in action!
"""

import requests
import time
import json
import os
from pathlib import Path
import webbrowser

# Configuration
FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:8000"
API_KEY = "omnibio-dev-key-12345"
TEST_FILE = "ST002091_AN003415.txt"

def print_header(title):
    """Print a fancy header"""
    print("\n" + "=" * 60)
    print(f"🔬 {title}")
    print("=" * 60)

def print_step(step, desc):
    """Print a step"""
    print(f"\n📋 Step {step}: {desc}")
    print("-" * 40)

def check_services():
    """Check if both services are running"""
    print_header("SERVICE STATUS CHECK")
    
    # Check backend
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Backend: {health['status']}")
            print(f"   Database: {'✅' if health['database_connected'] else '❌'}")
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return False
    
    # Check frontend
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend: Running")
        else:
            print(f"❌ Frontend error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend not running: {e}")
        return False
    
    return True

def demo_upload():
    """Demo file upload"""
    print_step(1, "File Upload Test")
    
    if not Path(TEST_FILE).exists():
        print(f"❌ Test file {TEST_FILE} not found")
        return None
    
    headers = {"X-API-Key": API_KEY}
    
    print(f"📤 Uploading: {TEST_FILE}")
    print(f"   Size: {Path(TEST_FILE).stat().st_size / 1024:.1f} KB")
    
    try:
        with open(TEST_FILE, 'rb') as f:
            files = {'file': (TEST_FILE, f, 'text/plain')}
            response = requests.post(f"{BACKEND_URL}/upload", headers=headers, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful!")
            print(f"   File ID: {result['file_id']}")
            print(f"   Type: {result['file_type']}")
            return result['file_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def demo_analysis(file_id):
    """Demo analysis creation and monitoring"""
    print_step(2, "Analysis Creation & Monitoring")
    
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "file_ids": [file_id],
        "analysis_types": ["statistical"],
        "project_name": "Live Demo Analysis"
    }
    
    print("🧪 Creating analysis...")
    try:
        response = requests.post(f"{BACKEND_URL}/analyze", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            analysis_id = result['analysis_id']
            print(f"✅ Analysis created: {analysis_id[:12]}...")
            
            # Monitor progress
            return monitor_analysis(analysis_id)
        else:
            print(f"❌ Analysis creation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def monitor_analysis(analysis_id):
    """Monitor analysis progress with real-time updates"""
    print("\n⏱️ Monitoring analysis progress...")
    headers = {"X-API-Key": API_KEY}
    
    start_time = time.time()
    
    for attempt in range(30):  # 30 attempts = 1 minute max
        try:
            response = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}", headers=headers)
            if response.status_code == 200:
                analysis = response.json()
                status = analysis['status']
                progress = analysis['progress']
                elapsed = time.time() - start_time
                
                # Progress bar
                bar_length = 30
                filled = int(bar_length * progress)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                print(f"\r   [{bar}] {progress:.1%} | {status} | {elapsed:.1f}s", end="", flush=True)
                
                if status == "completed":
                    print(f"\n✅ Analysis completed in {elapsed:.1f} seconds!")
                    return analysis_id
                elif status == "failed":
                    print(f"\n❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
                    return None
                
                time.sleep(2)
            else:
                print(f"\n❌ Failed to check status: {response.status_code}")
                return None
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            return None
    
    print("\n⏱️ Analysis taking longer than expected...")
    return analysis_id

def show_results(analysis_id):
    """Show analysis results"""
    print_step(3, "Results Visualization")
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        # Get analysis details
        response = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}", headers=headers)
        if response.status_code == 200:
            analysis = response.json()
            print("📊 Analysis Summary:")
            print(f"   Status: {analysis['status']}")
            print(f"   Progress: {analysis['progress']:.1%}")
            if analysis.get('started_at'):
                print(f"   Started: {analysis['started_at'][:19]}")
            if analysis.get('completed_at'):
                print(f"   Completed: {analysis['completed_at'][:19]}")
        
        # Check for generated files
        results_dir = Path(f"biomarker/results/{analysis_id}")
        if results_dir.exists():
            print(f"\n📁 Generated Files:")
            for file_path in results_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(results_dir)
                    size = file_path.stat().st_size
                    print(f"   📄 {rel_path} ({size:,} bytes)")
        
        # Try to get results via API
        try:
            response = requests.get(f"{BACKEND_URL}/analyses/{analysis_id}/results", headers=headers)
            if response.status_code == 200:
                results = response.json()
                print(f"\n📈 Results Preview:")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"   {key}: {len(value)} items")
                        else:
                            print(f"   {key}: {type(value).__name__}")
        except Exception as e:
            print(f"   API results not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Results error: {e}")
        return False

def open_frontend():
    """Open frontend in browser"""
    print_step(4, "Frontend Access")
    
    print("🌐 Opening OmniBio frontend...")
    print(f"   URL: {FRONTEND_URL}")
    print("   Login with API key: omnibio-dev-key-12345")
    
    try:
        webbrowser.open(FRONTEND_URL)
        print("✅ Browser opened")
    except Exception as e:
        print(f"⚠️ Could not open browser: {e}")
        print(f"   Please manually visit: {FRONTEND_URL}")

def main():
    """Run the live demo"""
    print_header("OMNIBIO LIVE DEMO")
    print("This demo will test your complete system with real data!")
    
    # Check services
    if not check_services():
        print("\n❌ Services not running. Please start:")
        print("   Backend: cd biomarker/api && python -m uvicorn main:app --reload --port 8000")
        print("   Frontend: cd omnibio-frontend && npm run dev")
        return
    
    # Run demo workflow
    file_id = demo_upload()
    if not file_id:
        print("\n❌ Demo failed at upload step")
        return
    
    analysis_id = demo_analysis(file_id)
    if not analysis_id:
        print("\n❌ Demo failed at analysis step")
        return
    
    show_results(analysis_id)
    
    # Open frontend
    open_frontend()
    
    # Summary
    print_header("DEMO COMPLETE")
    print("🎉 Your OmniBio system is working perfectly!")
    print("\n📋 What you can do next:")
    print("   1. Upload more files via the web interface")
    print("   2. Try different analysis types")
    print("   3. Download results for further analysis")
    print("   4. Explore the API documentation at /docs")
    
    print(f"\n🔗 Quick Links:")
    print(f"   Frontend: {FRONTEND_URL}")
    print(f"   API Docs: {BACKEND_URL}/docs")
    print(f"   Results: biomarker/results/{analysis_id}")

if __name__ == "__main__":
    main() 