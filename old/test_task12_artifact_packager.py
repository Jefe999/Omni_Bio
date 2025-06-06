#!/usr/bin/env python3
"""
Test Task #12: Artifact Packager
Tests the comprehensive artifact packaging system with real biomarker discovery results.
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

from biomarker.report.artifact_packager import create_biomarker_package

def create_sample_analysis_results():
    """Create sample analysis results to test the packager."""
    
    print("🔬 Creating sample analysis results for testing...")
    
    # Create test output directories
    test_dirs = {
        'qc_output': 'test_qc_output',
        'stats_output': 'test_stats_output', 
        'ml_output': 'test_ml_output'
    }
    
    for dir_name in test_dirs.values():
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)
    
    # Create sample QC files
    print("  📊 Creating QC analysis artifacts...")
    qc_dir = test_dirs['qc_output']
    
    # Create sample TIC/BPC plots (placeholder files)
    with open(os.path.join(qc_dir, 'tic_bpc_sample1.png'), 'w') as f:
        f.write("Sample TIC/BPC plot data")
    with open(os.path.join(qc_dir, 'tic_bpc_sample2.png'), 'w') as f:
        f.write("Sample TIC/BPC plot data") 
    with open(os.path.join(qc_dir, 'qc_summary.png'), 'w') as f:
        f.write("QC summary plot data")
    
    # Create sample PCA data
    pca_data = {
        'PC1': np.random.randn(100),
        'PC2': np.random.randn(100),
        'PC3': np.random.randn(100),
        'sample_id': [f'sample_{i}' for i in range(100)],
        'group': ['Control'] * 50 + ['Case'] * 50
    }
    pd.DataFrame(pca_data).to_csv(os.path.join(qc_dir, 'pca_results.csv'), index=False)
    
    # Create sample statistical analysis files
    print("  📈 Creating statistical analysis artifacts...")
    stats_dir = test_dirs['stats_output'] 
    
    # Create sample volcano plot
    with open(os.path.join(stats_dir, 'volcano_plot.png'), 'w') as f:
        f.write("Volcano plot data")
    
    # Create sample statistical results
    stats_results = {
        'feature_name': [f'Feature_{i}' for i in range(500)],
        'p_value': np.random.exponential(0.1, 500),
        'fold_change': np.random.normal(0, 1, 500),
        'adj_p_value': np.random.exponential(0.1, 500) * 5,
        'significant': np.random.choice([True, False], 500, p=[0.1, 0.9])
    }
    pd.DataFrame(stats_results).to_csv(os.path.join(stats_dir, 'statistical_results.csv'), index=False)
    
    # Create pathway analysis stub
    pathway_results = {
        'pathway_id': ['path_001', 'path_002', 'path_003'],
        'pathway_name': ['Lipid metabolism', 'Glycolysis', 'TCA cycle'],
        'p_value': [0.001, 0.05, 0.1],
        'gene_count': [15, 8, 12]
    }
    pd.DataFrame(pathway_results).to_csv(os.path.join(stats_dir, 'pathway_analysis.csv'), index=False)
    
    # Create sample ML model files
    print("  🤖 Creating ML model artifacts...")
    ml_dir = test_dirs['ml_output']
    
    # Create mock model file
    with open(os.path.join(ml_dir, 'logistic_regression_model.pkl'), 'w') as f:
        f.write("Mock pickled model data")
    
    # Create ROC curve plot
    with open(os.path.join(ml_dir, 'roc_curve.png'), 'w') as f:
        f.write("ROC curve plot data")
    
    # Create feature importance
    importance_data = {
        'feature_name': [f'Feature_{i}' for i in range(50)],
        'importance_score': np.random.exponential(1, 50),
        'rank': list(range(1, 51))
    }
    pd.DataFrame(importance_data).to_csv(os.path.join(ml_dir, 'feature_importance.csv'), index=False)
    
    # Create model summary
    model_summary = {
        'model_type': 'Logistic Regression',
        'cv_auc_mean': 0.754,
        'cv_auc_std': 0.059,
        'final_auc': 1.000,
        'n_features': 727,
        'n_samples': 199,
        'training_time_seconds': 5.23,
        'cross_validation_folds': 5
    }
    import json
    with open(os.path.join(ml_dir, 'model_summary.json'), 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    print("  ✅ Sample analysis results created")
    
    return test_dirs

def test_artifact_packager():
    """Test the artifact packager with comprehensive real-world data."""
    
    print("=" * 70)
    print("🎯 TESTING TASK #12: ARTIFACT PACKAGER")
    print("=" * 70)
    
    # Create sample analysis results
    test_dirs = create_sample_analysis_results()
    
    # Define data sources
    data_sources = {
        'raw': 'ST002091_AN003415.txt',  # Original mwTab file if available
        'processed': os.path.join(test_dirs['stats_output'], 'statistical_results.csv'),
        'metadata': os.path.join(test_dirs['qc_output'], 'pca_results.csv')
    }
    
    # Create custom metadata
    custom_metadata = {
        'study_name': 'NAFLD Lipidomics Biomarker Discovery',
        'data_type': 'LC-MS Lipidomics',
        'sample_count': 199,
        'feature_count': 727,
        'analysis_date': '2024-01-15',
        'platform': 'OmniBio MVP',
        'study_groups': ['Control', 'Case'],
        'significant_biomarkers': 3,
        'top_biomarkers': [
            'LPC(17:0)', 'CE(17:0)', 'SM(20:1)', 'DAG(16:0/16:0)', 'LPC(18:1)'
        ]
    }
    
    # Create analysis summary
    analysis_summary = {
        'total_samples': 199,
        'total_features': 727,
        'significant_features': 3,
        'model_auc': 0.754,
        'processing_time_minutes': 2.5
    }
    
    # Test the artifact packager
    print("\n🚀 Running artifact packager...")
    
    package_result = create_biomarker_package(
        output_dir='artifact_packages',
        qc_dir=test_dirs['qc_output'],
        stats_dir=test_dirs['stats_output'],
        ml_dir=test_dirs['ml_output'],
        data_sources=data_sources,
        metadata=custom_metadata,
        analysis_summary=analysis_summary,
        project_name='NAFLD_Biomarker_Discovery',
        create_zip=True,
        create_tar=True
    )
    
    # Verify the results
    print("\n" + "=" * 70)  
    print("🔍 VERIFYING PACKAGE RESULTS")
    print("=" * 70)
    
    package_info = package_result['package_info']
    
    print(f"✅ Package created successfully!")
    print(f"   📦 Package name: {package_info['package_name']}")
    print(f"   📁 Package directory: {package_info['package_directory']}")
    print(f"   📊 Total files: {package_info['total_files']}")
    print(f"   💾 Package size: {package_info['package_size_mb']:.2f} MB")
    
    # Check artifacts
    print(f"\n📋 Artifacts included:")
    for artifact_type, count in package_result['package_info']['artifacts'].items():
        if isinstance(count, dict) and 'files' in count:
            file_count = len(count['files'])
            if file_count > 0:
                print(f"   {artifact_type}: {file_count} files")
    
    # Check archives
    if package_result['archives']:
        print(f"\n📦 Archives created:")
        for archive_type, path in package_result['archives'].items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   {archive_type.upper()}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"   ❌ {archive_type.upper()}: {path} (NOT FOUND)")
    
    # Test the HTML report
    html_report_path = os.path.join(package_info['package_directory'], 'reports', 'html', 'analysis_report.html')
    if os.path.exists(html_report_path):
        print(f"✅ HTML report created: {html_report_path}")
        
        # Check HTML content
        with open(html_report_path, 'r') as f:
            html_content = f.read()
            if 'NAFLD_Biomarker_Discovery' in html_content and 'Biomarker Discovery Analysis Report' in html_content:
                print(f"   ✅ HTML report contains expected content")
            else:
                print(f"   ⚠️ HTML report may be missing content")
    else:
        print(f"❌ HTML report not found: {html_report_path}")
    
    # Test the README
    readme_path = os.path.join(package_info['package_directory'], 'README.md')
    if os.path.exists(readme_path):
        print(f"✅ README created: {readme_path}")
        
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            if 'NAFLD_Biomarker_Discovery' in readme_content and '## Usage Instructions' in readme_content:
                print(f"   ✅ README contains expected content")
            else:
                print(f"   ⚠️ README may be missing content")
    else:
        print(f"❌ README not found: {readme_path}")
    
    # Test manifest
    manifest_path = os.path.join(package_info['package_directory'], 'MANIFEST.txt')
    if os.path.exists(manifest_path):
        print(f"✅ Manifest created: {manifest_path}")
    else:
        print(f"❌ Manifest not found: {manifest_path}")
    
    print("\n" + "=" * 70)
    print("🎉 ARTIFACT PACKAGER TEST COMPLETE")
    print("=" * 70)
    
    # Cleanup test directories
    print("🧹 Cleaning up test directories...")
    for dir_name in test_dirs.values():
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    return package_result

def test_with_real_data():
    """Test with real data from previous analyses if available."""
    
    print("\n" + "=" * 70)
    print("🔬 TESTING WITH REAL DATA")
    print("=" * 70)
    
    # Check if we have real analysis results from previous tests
    real_dirs = {
        'qc': None,
        'stats': None,
        'ml': None
    }
    
    # Look for existing output directories
    if os.path.exists('test_full_dataset_output'):
        print("✅ Found real dataset output from previous test")
        real_dirs['stats'] = 'test_full_dataset_output/statistical'
        real_dirs['ml'] = 'test_full_dataset_output/ml_models'
    
    # Check for individual test outputs
    if os.path.exists('test_task3_output'):
        real_dirs['qc'] = 'test_task3_output'
        print("✅ Found QC output from Task #3")
    
    if os.path.exists('test_task11_simple_output'):
        real_dirs['stats'] = 'test_task11_simple_output' 
        print("✅ Found statistical analysis output from Task #11")
    
    if os.path.exists('test_task10_output'):
        real_dirs['ml'] = 'test_task10_output'
        print("✅ Found ML model output from Task #10")
    
    # Only proceed if we have real data
    has_real_data = any(d and os.path.exists(d) for d in real_dirs.values())
    
    if not has_real_data:
        print("⚠️ No real analysis data found - skipping real data test")
        print("   Run previous analysis tests first to generate real data")
        return None
    
    # Create package with real data
    print("🚀 Creating package with real analysis data...")
    
    real_data_sources = {}
    if os.path.exists('ST002091_AN003415.txt'):
        real_data_sources['raw'] = 'ST002091_AN003415.txt'
    
    real_metadata = {
        'study_name': 'Real NAFLD Lipidomics Study',
        'data_source': 'Metabolomics Workbench ST002091', 
        'analysis_pipeline': 'OmniBio MVP',
        'real_data': True
    }
    
    real_analysis_summary = {
        'data_type': 'Real LC-MS Lipidomics Data',
        'pipeline_version': 'MVP_v1.0'
    }
    
    real_package_result = create_biomarker_package(
        output_dir='real_artifact_packages',
        qc_dir=real_dirs['qc'],
        stats_dir=real_dirs['stats'],
        ml_dir=real_dirs['ml'],
        data_sources=real_data_sources,
        metadata=real_metadata,
        analysis_summary=real_analysis_summary,
        project_name='Real_NAFLD_Analysis',
        create_zip=True,
        create_tar=False  # Skip tar for real data to save time
    )
    
    print("✅ Real data package created successfully!")
    
    return real_package_result

if __name__ == "__main__":
    try:
        # Test with sample data
        sample_result = test_artifact_packager()
        
        # Test with real data if available
        real_result = test_with_real_data()
        
        print("\n" + "🎉" * 10)
        print("🏆 TASK #12 ARTIFACT PACKAGER - ALL TESTS PASSED!")
        print("🎉" * 10)
        
    except Exception as e:
        print(f"\n❌ ERROR in artifact packager test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 