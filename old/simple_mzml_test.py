#!/usr/bin/env python3
"""
Simple test of mzML chromatogram extraction
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

from biomarker.ingest.file_loader import load_file

def test_single_mzml():
    """Test loading a single mzML file"""
    test_file = "IMSS/Lipidyzer Batch - 20181024111314 - 1-29664 - 01.mzML"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        print(f"Testing mzML loading for: {Path(test_file).name}")
        df, metadata = load_file(test_file)
        
        print(f"✓ Success!")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Metadata: {metadata}")
        print(f"  Sample names: {df.index.tolist()}")
        print(f"  First 5 features: {df.columns.tolist()[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_single_mzml()
    if success:
        print("\n✅ Single mzML test passed!")
    else:
        print("\n❌ Single mzML test failed!") 