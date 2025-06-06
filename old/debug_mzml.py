#!/usr/bin/env python3
"""
Debug mzML file structure to understand why MS1 peaks are not found
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

def debug_mzml_file(file_path: str):
    """Debug an mzML file to understand its structure"""
    print(f"\n=== Debugging mzML file: {Path(file_path).name} ===")
    
    try:
        run = pymzml.run.Reader(str(file_path))
        
        spectrum_count = 0
        ms_levels = {}
        sample_spectra = []
        
        print("Examining first 10 spectra...")
        
        for i, spectrum in enumerate(run):
            spectrum_count += 1
            
            # Track MS levels
            ms_level = spectrum.ms_level
            if ms_level not in ms_levels:
                ms_levels[ms_level] = 0
            ms_levels[ms_level] += 1
            
            # Store first few spectra for detailed analysis
            if i < 5:
                peaks = spectrum.peaks("raw")
                rt = spectrum.scan_time_in_minutes()
                
                sample_spectra.append({
                    'index': i,
                    'ms_level': ms_level,
                    'n_peaks': len(peaks) if peaks is not None else 0,
                    'rt': rt,
                    'id': spectrum.ID if hasattr(spectrum, 'ID') else 'N/A'
                })
                
                print(f"  Spectrum {i}: MS{ms_level}, {len(peaks) if peaks is not None else 0} peaks, RT={rt}")
            
            # Limit for debugging
            if spectrum_count > 50:
                break
        
        print(f"\nSummary:")
        print(f"  Total spectra examined: {spectrum_count}")
        print(f"  MS levels found: {ms_levels}")
        
        # Check if we have MS1 data
        if 1 in ms_levels:
            print(f"  ✓ Found {ms_levels[1]} MS1 spectra")
        else:
            print(f"  ❌ No MS1 spectra found!")
            print(f"  Available MS levels: {list(ms_levels.keys())}")
        
        # Show detailed info for first few spectra
        print(f"\nDetailed spectrum info:")
        for spec_info in sample_spectra:
            print(f"  Spectrum {spec_info['index']}: MS{spec_info['ms_level']}, "
                  f"{spec_info['n_peaks']} peaks, RT={spec_info['rt']}, ID={spec_info['id']}")
        
        return ms_levels
        
    except Exception as e:
        print(f"❌ Error reading mzML file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Debug multiple mzML files"""
    imss_dir = Path("./IMSS")
    
    if not imss_dir.exists():
        print(f"❌ IMSS directory not found: {imss_dir}")
        return
    
    mzml_files = list(imss_dir.glob("*.mzML"))
    if len(mzml_files) == 0:
        print(f"❌ No mzML files found in {imss_dir}")
        return
    
    print(f"Found {len(mzml_files)} mzML files")
    
    # Debug first 3 files
    for i, mzml_file in enumerate(sorted(mzml_files)[:3]):
        result = debug_mzml_file(str(mzml_file))
        
        if result is None:
            print(f"Failed to process {mzml_file.name}")
        else:
            print(f"Successfully analyzed {mzml_file.name}")

if __name__ == '__main__':
    main() 