#!/usr/bin/env python3
"""
Test for Task #3 - TIC & BPC QC Plots
"""

import sys
import os
import shutil
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

def test_qc_plots():
    """Test QC plot generation functionality"""
    
    # Test file
    test_file = "ST002091_AN003415.txt"
    output_dir = "test_qc_output"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    # Clean up any existing output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        from biomarker.qc.qc_plots import generate_tic_bpc_plots, generate_qc_summary_plot
        print("✓ Successfully imported QC plotting modules")
        
        # Test 1: Load data
        print("\n--- Test 1: Data Loading ---")
        df, metadata = load_file(test_file)
        print(f"✓ Loaded data: {df.shape}")
        
        # Test 2: Generate individual QC plots
        print("\n--- Test 2: Individual QC Plots ---")
        plot_files = generate_tic_bpc_plots(df, output_dir, max_samples=3)
        
        if len(plot_files) == 0:
            print("❌ No plot files generated")
            return False
        
        print(f"✓ Generated {len(plot_files)} individual plots")
        
        # Verify files exist and have content
        for plot_file in plot_files:
            if not plot_file.exists():
                print(f"❌ Plot file missing: {plot_file}")
                return False
            
            if plot_file.stat().st_size == 0:
                print(f"❌ Plot file is empty: {plot_file}")
                return False
                
            print(f"✓ Verified plot: {plot_file.name} ({plot_file.stat().st_size} bytes)")
        
        # Test 3: Generate summary plot
        print("\n--- Test 3: Summary QC Plot ---")
        summary_file = generate_qc_summary_plot(df, output_dir, max_samples=5)
        
        if not summary_file.exists():
            print(f"❌ Summary plot missing: {summary_file}")
            return False
            
        if summary_file.stat().st_size == 0:
            print(f"❌ Summary plot is empty: {summary_file}")
            return False
            
        print(f"✓ Generated summary plot: {summary_file.name} ({summary_file.stat().st_size} bytes)")
        
        # Test 4: Verify output directory structure
        print("\n--- Test 4: Output Verification ---")
        output_path = Path(output_dir)
        all_files = list(output_path.glob("*.png"))
        
        print(f"✓ Total PNG files in output: {len(all_files)}")
        print(f"✓ Output directory: {output_path.absolute()}")
        
        # List all generated files
        print("Generated files:")
        for file in sorted(all_files):
            print(f"  - {file.name}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Task #3 - TIC & BPC QC Plots - COMPLETED!")
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
    output_dir = "test_qc_output"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up test output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")


if __name__ == '__main__':
    print("Testing Task #3 - TIC & BPC QC Plots")
    print("=" * 50)
    
    success = test_qc_plots()
    
    if success:
        print("\n✅ Task #3 is ready! Moving to next task...")
        
        # Ask user if they want to keep the test plots
        try:
            keep_plots = input("\nKeep test QC plots for inspection? (y/n): ").lower().strip()
            if keep_plots != 'y':
                cleanup_test_output()
        except:
            pass  # In case input is not available
            
        sys.exit(0)
    else:
        print("\n❌ Task #3 needs fixes before proceeding")
        cleanup_test_output()
        sys.exit(1) 