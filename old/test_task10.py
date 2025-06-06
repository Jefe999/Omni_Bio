#!/usr/bin/env python3
"""
Test for Task #10 - Model Training Pipeline
"""

import sys
import os
import shutil
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

def test_model_training():
    """Test model training pipeline functionality"""
    
    # Test file and output directory
    test_file = "ST002091_AN003415.txt"
    output_dir = "test_model_output"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    # Clean up any existing output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        from biomarker.ml.model_pipeline import train_models, prepare_ml_data
        print("✓ Successfully imported model training modules")
        
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
            # Create dummy labels (first half = Case, second half = Control)
            import pandas as pd
            n_samples = len(df)
            labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                             index=df.index, name='Group')
            print(f"✓ Created dummy labels: {labels.value_counts().to_dict()}")
        
        # Test 3: Data preparation
        print("\n--- Test 3: Data Preparation ---")
        X, y = prepare_ml_data(df, labels)
        print(f"✓ Prepared data for ML: X={X.shape}, y={y.shape}")
        
        # Test 4: Model training
        print("\n--- Test 4: Model Training ---")
        results = train_models(df, labels, output_dir, cv_folds=3)  # Use 3 folds for faster testing
        
        if len(results) == 0:
            print("❌ No models were trained successfully")
            return False
        
        print(f"✓ Trained {len(results)} model(s)")
        
        # Test 5: Verify artifacts
        print("\n--- Test 5: Artifact Verification ---")
        output_path = Path(output_dir)
        
        if not output_path.exists():
            print(f"❌ Output directory not created: {output_path}")
            return False
            
        total_artifacts = 0
        for model_name, model_data in results.items():
            print(f"\n  {model_name.upper()}:")
            artifacts = model_data['artifacts']
            
            # Check each artifact type
            for artifact_type, artifact_path in artifacts.items():
                if not artifact_path.exists():
                    print(f"    ❌ Missing {artifact_type}: {artifact_path}")
                    return False
                    
                if artifact_path.stat().st_size == 0:
                    print(f"    ❌ Empty {artifact_type}: {artifact_path}")
                    return False
                    
                print(f"    ✓ {artifact_type}: {artifact_path.name} ({artifact_path.stat().st_size} bytes)")
                total_artifacts += 1
            
            # Check performance
            model_results = model_data['results']
            final_auc = model_results['final_auc']
            print(f"    ✓ Final AUC: {final_auc:.3f}")
            
            # Verify AUC is reasonable (should be > 0.5 for any real model)
            if final_auc <= 0.5:
                print(f"    ❌ AUC too low: {final_auc:.3f} (should be > 0.5)")
                return False
        
        print(f"\n✓ Total artifacts generated: {total_artifacts}")
        
        # Test 6: Check specific files
        print("\n--- Test 6: File Content Checks ---")
        all_files = list(output_path.glob("*"))
        expected_extensions = ['.pkl', '.txt', '.json', '.png']  # model files, summaries, plots
        
        found_extensions = set()
        for file in all_files:
            found_extensions.add(file.suffix)
        
        print(f"✓ File types found: {sorted(found_extensions)}")
        
        # Should have at least JSON (summaries) and PNG (ROC plots)
        if '.json' not in found_extensions:
            print("❌ No JSON summary files found")
            return False
            
        if '.png' not in found_extensions:
            print("❌ No PNG plot files found")
            return False
        
        print(f"✓ Generated {len(all_files)} total files")
        
        # List all files
        print("Generated files:")
        for file in sorted(all_files):
            print(f"  - {file.name}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Task #10 - Model Training Pipeline - COMPLETED!")
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
    output_dir = "test_model_output"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up test output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")


if __name__ == '__main__':
    print("Testing Task #10 - Model Training Pipeline")
    print("=" * 50)
    
    success = test_model_training()
    
    if success:
        print("\n✅ Task #10 is ready! Moving to next task...")
        
        # Ask user if they want to keep the test results
        try:
            keep_results = input("\nKeep test model results for inspection? (y/n): ").lower().strip()
            if keep_results != 'y':
                cleanup_test_output()
        except:
            pass  # In case input is not available
            
        sys.exit(0)
    else:
        print("\n❌ Task #10 needs fixes before proceeding")
        cleanup_test_output()
        sys.exit(1) 