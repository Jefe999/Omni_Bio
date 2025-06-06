#!/usr/bin/env python3
"""
Debug chromatogram extraction from SRM/MRM mzML files
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

try:
    import pymzml
    print("✓ pymzml is available")
except ImportError:
    print("❌ pymzml not available")
    sys.exit(1)

def test_chromatogram_extraction(file_path: str):
    """Test different methods to extract chromatogram data"""
    print(f"\n=== Testing chromatogram extraction: {Path(file_path).name} ===")
    
    try:
        run = pymzml.run.Reader(str(file_path))
        
        print("Method 1: Direct iteration through run")
        count = 0
        for item in run:
            count += 1
            print(f"  Item {count}: {type(item)}, has ID: {hasattr(item, 'ID')}, has id: {hasattr(item, 'id')}")
            if count > 5:  # Just check first few
                break
        
        print(f"Total items found: {count}")
        
        # Try to access chromatograms differently
        print("\nMethod 2: Check if run has chromatograms attribute")
        if hasattr(run, 'chromatograms'):
            print("  ✓ Run has chromatograms attribute")
            try:
                chroms = run.chromatograms()
                print(f"  Found {len(chroms)} chromatograms")
            except Exception as e:
                print(f"  Error accessing chromatograms: {e}")
        else:
            print("  ❌ Run does not have chromatograms attribute")
        
        # Try alternative approach
        print("\nMethod 3: Check run info")
        run_info = dir(run)
        chrom_methods = [method for method in run_info if 'chrom' in method.lower()]
        print(f"  Chromatogram-related methods: {chrom_methods}")
        
        # Try to access run info
        print("\nMethod 4: Run info")
        try:
            info = run.info
            print(f"  Run info: {info}")
        except:
            print("  No run info available")
        
        # Reset and try a different approach - access via index
        print("\nMethod 5: Try accessing by index")
        try:
            run = pymzml.run.Reader(str(file_path))
            
            # Try to get a specific chromatogram by ID
            try:
                # Get TIC chromatogram (we saw this in the XML)
                tic_chrom = run['TIC']
                print(f"  ✓ Found TIC chromatogram: {type(tic_chrom)}")
                
                # Check what attributes it has
                attrs = [attr for attr in dir(tic_chrom) if not attr.startswith('_')]
                print(f"  TIC chromatogram attributes: {attrs[:10]}...")  # Show first 10
                
                # Try to get time and intensity data
                if hasattr(tic_chrom, 'time'):
                    times = tic_chrom.time
                    print(f"  Times: {len(times) if times is not None else 'None'} points")
                
                if hasattr(tic_chrom, 'i'):
                    intensities = tic_chrom.i
                    print(f"  Intensities: {len(intensities) if intensities is not None else 'None'} points")
                
                if hasattr(tic_chrom, 'peaks'):
                    peaks = tic_chrom.peaks()
                    print(f"  Peaks: {len(peaks) if peaks is not None else 'None'} points")
                    
            except Exception as e:
                print(f"  Error accessing TIC: {e}")
                
        except Exception as e:
            print(f"  Error with index access: {e}")
        
    except Exception as e:
        print(f"❌ Error reading mzML file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Test chromatogram extraction"""
    test_file = "IMSS/Lipidyzer Batch - 20181024111314 - 1-29664 - 01.mzML"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    test_chromatogram_extraction(test_file)

if __name__ == '__main__':
    main() 