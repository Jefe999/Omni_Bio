#!/usr/bin/env python3
"""
Test script for batch mzML processing with improved formatting
"""

import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from biomarker.io.batch_mzml_processor import (
    process_mzml_batch, 
    create_case_control_labels, 
    save_feature_matrix_as_mwtab
)

def test_batch_processing():
    """Test batch processing with improved mwTab format"""
    print("🧪 Testing Improved Batch mzML Processing")
    print("=" * 50)
    
    # Define paths
    mzml_dir = Path("old/IMSS")
    output_dir = Path("mzml_test_output")
    
    if not mzml_dir.exists():
        print(f"❌ mzML directory not found: {mzml_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find mzML files (limit to 10 for testing)
    mzml_files = list(mzml_dir.glob("*.mzML"))[:10]
    
    if not mzml_files:
        print(f"❌ No mzML files found in {mzml_dir}")
        return
    
    print(f"📁 Found {len(mzml_files)} mzML files for processing")
    
    # Process batch
    feature_matrix, metadata = process_mzml_batch(mzml_files)
    
    print(f"✅ Processed feature matrix: {feature_matrix.shape}")
    
    # Create labels based on filename patterns
    labels = create_case_control_labels(
        sample_names=feature_matrix.index.tolist(),
        case_patterns=['20181107'],     # Batch 2 = Cases  
        control_patterns=['20181106']   # Batch 1 = Controls
    )
    
    print("🏷️ Label distribution:")
    label_counts = {}
    for label in labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in label_counts.items():
        print(f"   {label}: {count} samples")
    
    # Save with improved formatting
    output_file = output_dir / "lipidomics_batch_test_improved.txt"
    save_feature_matrix_as_mwtab(
        feature_matrix, 
        labels, 
        str(output_file),
        study_title="IMSS Lipidomics Batch Test - Improved Format"
    )
    
    print(f"\n📊 Results Summary:")
    print(f"   Samples: {feature_matrix.shape[0]}")
    print(f"   Features: {feature_matrix.shape[1]}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

def test_single_mzml_upload():
    """Test uploading a single mzML file directly"""
    print("\n🧪 Testing Single mzML File Upload Capability")
    print("=" * 50)
    
    # Test if biomarker platform can handle raw mzML
    try:
        from biomarker.io.file_loader import detect_file_type, load_file
        
        # Find a single mzML file
        mzml_dir = Path("old/IMSS")
        mzml_files = list(mzml_dir.glob("*.mzML"))
        
        if mzml_files:
            test_file = mzml_files[0]
            print(f"📄 Testing file: {test_file.name}")
            
            # Test detection
            file_type = detect_file_type(str(test_file))
            print(f"✅ File type detected: {file_type}")
            
            # Test loading
            df, metadata = load_file(str(test_file))
            print(f"✅ Loaded successfully: {df.shape[0]} samples × {df.shape[1]} features")
            print(f"   Sample name: {df.index[0]}")
            print(f"   First few features: {list(df.columns[:5])}")
            
            print(f"\n💡 RECOMMENDATION: You can upload {test_file.name} directly to the biomarker platform!")
            
        else:
            print("❌ No mzML files found for testing")
            
    except Exception as e:
        print(f"❌ Error testing direct mzML upload: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_processing()
    test_single_mzml_upload() 