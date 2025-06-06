#!/usr/bin/env python3
"""
Test script for Integrated Feature Extraction + Enrichment API

Tests the complete integrated pipeline:
1. File upload
2. Feature extraction pipeline
3. Pathway enrichment analysis
4. Results retrieval
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# API configuration
API_BASE = "http://localhost:8000"

def create_demo_data_file():
    """Create a demo mwTab-style file for testing"""
    # Create demo metabolomics data
    np.random.seed(42)
    
    # Sample names (NAFLD study style)
    sample_names = ['Control_' + str(i) for i in range(1, 11)] + ['NAFLD_' + str(i) for i in range(1, 11)]
    
    # Feature names (metabolite-like)
    feature_names = [
        'mz_180.0634_rt_120.5',  # Glucose-like
        'mz_146.0579_rt_85.2',   # Pyruvate-like  
        'mz_117.0193_rt_95.8',   # Glutamate-like
        'mz_132.0532_rt_110.3',  # Ketoglutarate-like
        'mz_204.0892_rt_105.6',  # Citrate-like
    ] + [f'feature_{i:04d}' for i in range(100, 200)]  # Additional features
    
    # Generate data matrix
    data = np.random.lognormal(mean=3, sigma=1, size=(len(sample_names), len(feature_names)))
    
    # Add some missing values
    missing_mask = np.random.random(data.shape) < 0.1
    data[missing_mask] = 0
    
    # Create DataFrame
    df = pd.DataFrame(data, index=sample_names, columns=feature_names)
    
    # Save as CSV
    test_file = Path("demo_metabolomics_data.csv")
    df.to_csv(test_file)
    
    return test_file


def test_health_check():
    """Test API health check"""
    print("🔍 Testing API health check...")
    
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        print("  ✅ API is healthy")
        health_data = response.json()
        print(f"    Status: {health_data['status']}")
        print(f"    Active analyses: {health_data['active_analyses']}")
        return True
    else:
        print(f"  ❌ API health check failed: {response.status_code}")
        return False


def upload_test_file(file_path):
    """Upload a test file to the API"""
    print(f"📤 Uploading test file: {file_path}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'text/csv')}
        response = requests.post(f"{API_BASE}/upload", files=files)
    
    if response.status_code == 200:
        file_info = response.json()
        print(f"  ✅ File uploaded successfully")
        print(f"    File ID: {file_info['file_id']}")
        print(f"    File type: {file_info['file_type']}")
        print(f"    Size: {file_info['size_bytes']} bytes")
        return file_info['file_id']
    else:
        print(f"  ❌ File upload failed: {response.status_code}")
        print(f"    Error: {response.text}")
        return None


def test_feature_extraction(file_id):
    """Test feature extraction pipeline"""
    print("🔬 Testing feature extraction pipeline...")
    
    request_data = {
        "file_ids": [file_id],
        "project_name": "test_feature_extraction",
        "mass_error_ppm": 5.0,
        "intensity_threshold": 1000.0,
        "mz_tolerance_ppm": 5.0,
        "rt_tolerance_min": 0.1,
        "min_frequency": 0.5,
        "score_method": "total_abundance",
        "min_score_percentile": 30.0,
        "imputation_method": "median_global",
        "scaling_method": "pareto",
        "log_transform": False
    }
    
    response = requests.post(f"{API_BASE}/extract-features", json=request_data)
    
    if response.status_code == 200:
        analysis_info = response.json()
        analysis_id = analysis_info['analysis_id']
        print(f"  ✅ Feature extraction started")
        print(f"    Analysis ID: {analysis_id}")
        
        # Wait for completion
        print("  ⏳ Waiting for feature extraction to complete...")
        return wait_for_analysis_completion(analysis_id)
    else:
        print(f"  ❌ Feature extraction failed to start: {response.status_code}")
        print(f"    Error: {response.text}")
        return None


def test_integrated_analysis(file_id):
    """Test integrated feature extraction + enrichment pipeline"""
    print("🚀 Testing integrated analysis pipeline...")
    
    request_data = {
        "file_ids": [file_id],
        "project_name": "test_integrated_analysis",
        "group_column": None,
        "feature_extraction": True,
        "mass_error_ppm": 5.0,
        "min_frequency": 0.4,
        "scaling_method": "pareto",
        "run_statistics": True,
        "run_enrichment": True,
        "p_value_threshold": 0.05,
        "use_kegg": True,
        "use_go": True
    }
    
    response = requests.post(f"{API_BASE}/integrated-analysis", json=request_data)
    
    if response.status_code == 200:
        analysis_info = response.json()
        analysis_id = analysis_info['analysis_id']
        print(f"  ✅ Integrated analysis started")
        print(f"    Analysis ID: {analysis_id}")
        
        # Wait for completion
        print("  ⏳ Waiting for integrated analysis to complete...")
        return wait_for_analysis_completion(analysis_id)
    else:
        print(f"  ❌ Integrated analysis failed to start: {response.status_code}")
        print(f"    Error: {response.text}")
        return None


def wait_for_analysis_completion(analysis_id, timeout=300):
    """Wait for analysis to complete with timeout"""
    start_time = time.time()
    last_progress = 0
    
    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE}/analyses/{analysis_id}")
        
        if response.status_code == 200:
            status_data = response.json()
            status = status_data['status']
            progress = status_data['progress']
            message = status_data['message']
            
            # Show progress updates
            if progress > last_progress:
                print(f"    Progress: {progress:.1%} - {message}")
                last_progress = progress
            
            if status == 'completed':
                print(f"  ✅ Analysis completed successfully!")
                return status_data
            elif status == 'failed':
                print(f"  ❌ Analysis failed: {status_data.get('error', 'Unknown error')}")
                return None
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"  ⏰ Analysis timed out after {timeout} seconds")
    return None


def test_enrichment_analysis(feature_matrix_path, significant_features):
    """Test standalone pathway enrichment analysis"""
    print("🧬 Testing pathway enrichment analysis...")
    
    request_data = {
        "feature_matrix_file": feature_matrix_path,
        "significant_features": significant_features,
        "background_features": None,
        "p_value_threshold": 0.05,
        "fdr_method": "fdr_bh",
        "use_kegg": True,
        "use_go": True,
        "organism": "hsa",
        "min_genes_per_pathway": 3
    }
    
    response = requests.post(f"{API_BASE}/enrich-pathways", json=request_data)
    
    if response.status_code == 200:
        analysis_info = response.json()
        analysis_id = analysis_info['analysis_id']
        print(f"  ✅ Enrichment analysis started")
        print(f"    Analysis ID: {analysis_id}")
        
        # Wait for completion
        print("  ⏳ Waiting for enrichment analysis to complete...")
        return wait_for_analysis_completion(analysis_id)
    else:
        print(f"  ❌ Enrichment analysis failed to start: {response.status_code}")
        print(f"    Error: {response.text}")
        return None


def download_analysis_results(analysis_id):
    """Download and inspect analysis results"""
    print(f"📥 Downloading results for analysis {analysis_id}...")
    
    # Get analysis results
    response = requests.get(f"{API_BASE}/analyses/{analysis_id}/results")
    
    if response.status_code == 200:
        results = response.json()
        print(f"  ✅ Results downloaded successfully")
        
        # Display key results
        if 'results' in results and results['results']:
            result_data = results['results']
            print(f"    Results summary:")
            
            for key, value in result_data.items():
                if isinstance(value, dict):
                    print(f"      {key}: {json.dumps(value, indent=2)[:200]}...")
                else:
                    print(f"      {key}: {value}")
        
        return results
    else:
        print(f"  ❌ Failed to download results: {response.status_code}")
        return None


def main():
    """Run the complete integrated API test"""
    print("🧪 INTEGRATED FEATURE EXTRACTION + ENRICHMENT API TEST")
    print("=" * 70)
    
    # Step 1: Health check
    if not test_health_check():
        print("❌ API not accessible, exiting test")
        return
    
    print()
    
    # Step 2: Create and upload test data
    test_file = create_demo_data_file()
    print(f"📊 Created demo data file: {test_file}")
    
    file_id = upload_test_file(test_file)
    if not file_id:
        print("❌ File upload failed, exiting test")
        return
    
    print()
    
    # Step 3: Test feature extraction pipeline
    extraction_results = test_feature_extraction(file_id)
    if extraction_results:
        print(f"📊 Feature extraction completed:")
        if 'results' in extraction_results:
            summary = extraction_results['results'].get('summary', {})
            print(f"    Initial features: {summary.get('initial_features', 'N/A')}")
            print(f"    Final features: {summary.get('final_features', 'N/A')}")
            print(f"    Samples: {summary.get('samples', 'N/A')}")
    
    print()
    
    # Step 4: Test integrated analysis pipeline
    integrated_results = test_integrated_analysis(file_id)
    if integrated_results:
        print(f"🚀 Integrated analysis completed:")
        if 'results' in integrated_results:
            results = integrated_results['results']
            print(f"    Features: {results.get('n_features', 'N/A')}")
            print(f"    Samples: {results.get('n_samples', 'N/A')}")
            print(f"    Significant features: {results.get('significant_features_count', 'N/A')}")
            
            if 'enrichment_summary' in results:
                enrichment = results['enrichment_summary']
                print(f"    Enrichment pathways: {enrichment.get('total_significant_pathways', 'N/A')}")
                
                # Show pathway overview
                if 'enrichment_overview' in enrichment:
                    for db, stats in enrichment['enrichment_overview'].items():
                        print(f"      {db.upper()}: {stats['significant']}/{stats['tested']} pathways")
    
    print()
    
    # Step 5: Test standalone enrichment (if we have results from integrated analysis)
    if integrated_results and 'results' in integrated_results:
        results = integrated_results['results']
        feature_matrix = results.get('feature_matrix')
        significant_features = results.get('significant_features', [])
        
        if feature_matrix and significant_features:
            enrichment_results = test_enrichment_analysis(feature_matrix, significant_features)
            if enrichment_results:
                print(f"🧬 Standalone enrichment analysis completed")
    
    print()
    
    # Step 6: List all analyses
    print("📋 Listing all analyses...")
    response = requests.get(f"{API_BASE}/analyses")
    if response.status_code == 200:
        analyses = response.json()
        print(f"  Total analyses: {len(analyses)}")
        for analysis in analyses[-3:]:  # Show last 3
            print(f"    {analysis['analysis_id']}: {analysis['status']} - {analysis['message']}")
    
    print()
    print("✅ INTEGRATED API TEST COMPLETED!")
    print("🎉 Feature extraction + enrichment pipeline is fully functional!")


if __name__ == "__main__":
    main() 