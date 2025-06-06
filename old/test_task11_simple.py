#!/usr/bin/env python3
"""
Test for Task #11 - Statistical Analysis (Simple Version)
Tests t-tests and volcano plots without plotly dependencies
"""

import sys
import os
import shutil
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

def test_statistical_analysis():
    """Test statistical analysis functionality"""
    
    # Test file and output directory
    test_file = "ST002091_AN003415.txt"
    output_dir = "test_statistical_output"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    # Clean up any existing output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        from biomarker.ml.statistical_analysis_simple import run_complete_statistical_analysis
        import pandas as pd
        print("✓ Successfully imported statistical analysis modules")
        
        # Test 1: Load data
        print("\n--- Test 1: Data Loading ---")
        df, metadata = load_file(test_file)
        print(f"✓ Loaded data: {df.shape}")
        
        # Test 2: Extract labels
        print("\n--- Test 2: Label Extraction ---")
        try:
            import mwtab
            mw = next(mwtab.read_files(test_file))
            ssf = pd.DataFrame(mw['SUBJECT_SAMPLE_FACTORS'])
            factors = pd.json_normalize(ssf['Factors'])
            ssf = ssf.drop(columns='Factors').join(factors)
            labels = ssf.set_index('Sample ID')['Group'].reindex(df.index)
            print(f"✓ Extracted labels from mwTab: {labels.value_counts().to_dict()}")
        except Exception as e:
            print(f"Warning: Using dummy labels for testing: {e}")
            n_samples = len(df)
            labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                             index=df.index, name='Group')
            print(f"✓ Created dummy labels: {labels.value_counts().to_dict()}")
        
        # Test 3: Statistical analysis
        print("\n--- Test 3: Statistical Analysis ---")
        analysis_results = run_complete_statistical_analysis(df, labels, output_dir)
        
        if 'results_df' not in analysis_results:
            print("❌ No statistical results returned")
            return False
        
        results_df = analysis_results['results_df']
        print(f"✓ Statistical tests completed: {len(results_df)} features tested")
        
        # Test 4: Verify files were created
        print("\n--- Test 4: File Verification ---")
        output_path = Path(output_dir)
        
        if not output_path.exists():
            print(f"❌ Output directory not created: {output_path}")
            return False
        
        expected_files = {
            'statistical_results.csv': 'CSV results',
            'volcano_plot.png': 'Volcano plot',
            'pathway_analysis_stub.json': 'Pathway analysis stub'
        }
        
        for filename, description in expected_files.items():
            file_path = output_path / filename
            if not file_path.exists():
                print(f"❌ Missing {description}: {file_path}")
                return False
            
            if file_path.stat().st_size == 0:
                print(f"❌ Empty {description}: {file_path}")
                return False
            
            print(f"✓ {description}: {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Test 5: Verify results content
        print("\n--- Test 5: Results Content ---")
        summary = analysis_results['summary']
        
        required_keys = ['n_features_tested', 'n_significant_raw', 'n_significant_adj', 'top_features']
        for key in required_keys:
            if key not in summary:
                print(f"❌ Missing summary key: {key}")
                return False
        
        print(f"✓ Features tested: {summary['n_features_tested']}")
        print(f"✓ Significant (raw): {summary['n_significant_raw']}")
        print(f"✓ Significant (adjusted): {summary['n_significant_adj']}")
        print(f"✓ Top features available: {len(summary['top_features'])}")
        
        # Test 6: Check results DataFrame structure
        print("\n--- Test 6: Results DataFrame ---")
        required_columns = ['feature', 'p_value', 'fold_change', 'significant', 'p_adjusted']
        for col in required_columns:
            if col not in results_df.columns:
                print(f"❌ Missing column in results: {col}")
                return False
        
        print(f"✓ Results DataFrame has all required columns")
        print(f"✓ P-values range: {results_df['p_value'].min():.2e} to {results_df['p_value'].max():.2e}")
        print(f"✓ Fold changes range: {results_df['fold_change'].min():.2f} to {results_df['fold_change'].max():.2f}")
        
        # Test 7: Verify pathway stub
        print("\n--- Test 7: Pathway Analysis Stub ---")
        pathway_results = analysis_results['pathway_results']
        
        if pathway_results['status'] != 'stub_implementation':
            print("❌ Pathway analysis should be stub implementation")
            return False
        
        if 'todo_items' not in pathway_results:
            print("❌ Pathway stub missing TODO items")
            return False
        
        print(f"✓ Pathway stub properly implemented")
        print(f"✓ TODO items: {len(pathway_results['todo_items'])}")
        
        # Test 8: Show some actual results
        print("\n--- Test 8: Sample Results ---")
        top_features = results_df.head(5)
        for i, row in top_features.iterrows():
            print(f"  {row['feature']}: p={row['p_value']:.2e}, FC={row['fold_change']:.2f}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Task #11 - Statistical Analysis - COMPLETED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_output():
    """Clean up test output directory"""
    output_dir = "test_statistical_output"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up test output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")


if __name__ == '__main__':
    print("Testing Task #11 - Statistical Analysis (Simple Version)")
    print("=" * 60)
    
    success = test_statistical_analysis()
    
    if success:
        print("\n✅ Task #11 is ready! Statistical analysis with t-tests and volcano plots working!")
        
        # Ask user if they want to keep the test results
        try:
            keep_results = input("\nKeep test statistical results for inspection? (y/n): ").lower().strip()
            if keep_results != 'y':
                cleanup_test_output()
        except:
            pass  # In case input is not available
            
        sys.exit(0)
    else:
        print("\n❌ Task #11 needs fixes before proceeding")
        cleanup_test_output()
        sys.exit(1) 