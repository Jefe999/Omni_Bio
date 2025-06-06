#!/usr/bin/env python3
"""
Test for Task #1 - Unified File Loader
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

def test_file_loader():
    """Test the file loader with the existing mwTab file"""
    
    # Test file existence first
    test_file = "ST002091_AN003415.txt"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    print(f"✓ Found test file: {test_file}")
    
    try:
        # Import our file loader
        from biomarker.ingest.file_loader import detect_file_type, load_file, FileTypeError
        print("✓ Successfully imported file loader modules")
        
        # Test 1: File type detection
        print("\n--- Test 1: File Type Detection ---")
        file_type = detect_file_type(test_file)
        print(f"✓ Detected file type: {file_type}")
        
        if file_type != 'mwtab':
            print(f"❌ Expected 'mwtab', got '{file_type}'")
            return False
        
        # Test 2: File loading
        print("\n--- Test 2: File Loading ---")
        df, metadata = load_file(test_file)
        print(f"✓ Loaded data successfully")
        print(f"  Data shape: {df.shape}")
        print(f"  File type: {metadata['file_type']}")
        print(f"  Study ID: {metadata.get('study_id', 'N/A')}")
        print(f"  Analysis ID: {metadata.get('analysis_id', 'N/A')}")
        
        # Test 3: Data validation
        print("\n--- Test 3: Data Validation ---")
        
        if df.shape[0] == 0:
            print("❌ No samples found in data")
            return False
        print(f"✓ Found {df.shape[0]} samples")
        
        if df.shape[1] == 0:
            print("❌ No features found in data")
            return False
        print(f"✓ Found {df.shape[1]} features")
        
        # Check first few samples and features
        print(f"✓ First 5 samples: {list(df.index[:5])}")
        print(f"✓ First 5 features: {list(df.columns[:5])}")
        
        # Check data types
        if not all(df.dtypes.apply(lambda x: x.kind in 'bifc')):  # numeric types
            non_numeric = df.dtypes[~df.dtypes.apply(lambda x: x.kind in 'bifc')]
            print(f"Warning: Found non-numeric columns: {list(non_numeric.index[:5])}")
        else:
            print("✓ All data appears to be numeric")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Task #1 - Unified File Loader - COMPLETED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the biomarker package structure is correct")
        return False
        
    except FileTypeError as e:
        print(f"❌ File type error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Testing Task #1 - Unified File Loader")
    print("=" * 50)
    
    success = test_file_loader()
    
    if success:
        print("\n✅ Task #1 is ready! Moving to next task...")
        sys.exit(0)
    else:
        print("\n❌ Task #1 needs fixes before proceeding")
        sys.exit(1) 