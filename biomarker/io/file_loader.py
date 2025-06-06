#!/usr/bin/env python3
"""
Unified file loader for OmniBio MVP
Supports both mwTab and mzML files with auto-detection
Complete implementation of Task #1 from scoping document
"""

import os
import sys
from pathlib import Path
from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np
import warnings

# Import mwtab with proper error handling
try:
    import mwtab
    MWTAB_AVAILABLE = True
except ImportError:
    MWTAB_AVAILABLE = False
    print("Warning: mwtab not available. Please install with: pip install mwtab")

# Import pymzml with proper error handling
try:
    import pymzml
    PYMZML_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    PYMZML_AVAILABLE = False
    print(f"Warning: pymzml not available. mzML support disabled. Error: {e}")
    print("To enable mzML support, try: pip install pymzml")


class FileTypeError(Exception):
    """Raised when file type cannot be determined or is unsupported"""
    pass


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect file type based on extension and content
    
    Args:
        file_path: Path to the input file
        
    Returns:
        File type: 'mwtab', 'mzml_centroid', 'mzml_profile'
        
    Raises:
        FileTypeError: If file type cannot be determined
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileTypeError(f"File does not exist: {file_path}")
    
    # Check file extension
    ext = file_path.suffix.lower()
    
    if ext == '.txt':
        # Check if it's an mwTab file by looking for characteristic headers
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            
            # Look for mwTab characteristic markers
            mwtab_markers = ['#METABOLOMICS WORKBENCH', 'VERSION', 'CREATED_ON', 'MS_METABOLITE_DATA']
            
            content = ' '.join(first_lines).upper()
            if any(marker in content for marker in mwtab_markers):
                return 'mwtab'
            else:
                raise FileTypeError(f"Text file does not appear to be mwTab format: {file_path}")
                
        except Exception as e:
            raise FileTypeError(f"Error reading text file {file_path}: {e}")
    
    elif ext in ['.mzml', '.mzxml']:
        if not PYMZML_AVAILABLE:
            raise FileTypeError("pymzml not available for mzML file processing. Install with: pip install pymzml")
            
        try:
            # Quick check to determine if centroid or profile
            run = pymzml.run.Reader(str(file_path))
            
            # Check first few spectra
            profile_indicators = 0
            centroid_indicators = 0
            spectra_checked = 0
            
            for spectrum in run:
                if spectrum.ms_level == 1 and spectra_checked < 5:
                    spectra_checked += 1
                    
                    # Check if spectrum has centroided attribute
                    if hasattr(spectrum, 'centroided'):
                        if spectrum.centroided:
                            centroid_indicators += 1
                        else:
                            profile_indicators += 1
                    else:
                        # Heuristic: check peak density
                        peaks = spectrum.peaks("raw")
                        if len(peaks) > 0:
                            # Profile data typically has many more points
                            if len(peaks) > 10000:
                                profile_indicators += 1
                            else:
                                centroid_indicators += 1
                
                if spectra_checked >= 5:
                    break
            
            # Determine type based on majority
            if centroid_indicators > profile_indicators:
                return 'mzml_centroid'
            else:
                return 'mzml_profile'
            
        except Exception as e:
            raise FileTypeError(f"Error reading mzML file {file_path}: {e}")
    
    else:
        raise FileTypeError(f"Unsupported file extension: {ext}. Supported: .txt (mwTab), .mzml, .mzxml")


def extract_ms1_data_from_mzml(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Extract data from mzML file for feature matrix generation
    Handles both MS1 spectra and SRM/MRM chromatograms
    
    Args:
        file_path: Path to mzML file
        
    Returns:
        Tuple of (feature_matrix_dataframe, metadata_dict)
    """
    if not PYMZML_AVAILABLE:
        raise FileTypeError("pymzml not available for mzML processing")
    
    try:
        run = pymzml.run.Reader(str(file_path))
        
        print(f"Processing mzML file: {Path(file_path).name}")
        
        # Check if we have chromatogram data
        chromatogram_count = run.info.get('chromatogram_count', 0)
        offset_dict = run.info.get('offset_dict', {})
        
        # If chromatogram_count is 0 but we have offset_dict, count from offset_dict
        if chromatogram_count == 0 and len(offset_dict) > 0:
            chromatogram_count = len(offset_dict)
        
        if chromatogram_count > 0 and len(offset_dict) > 0:
            print(f"  Found {chromatogram_count} chromatograms (SRM/MRM data)")
            
            # Extract chromatogram data
            chromatogram_data = []
            
            for chrom_id in offset_dict.keys():
                try:
                    # Access chromatogram by ID
                    chromatogram = run[chrom_id]
                    
                    # Get time and intensity arrays
                    times = chromatogram.time if hasattr(chromatogram, 'time') else None
                    intensities = chromatogram.i if hasattr(chromatogram, 'i') else None
                    
                    if times is not None and intensities is not None and len(intensities) > 0:
                        # Use AUC (area under curve) as the feature value
                        auc = sum(intensities) if len(intensities) > 0 else 0.0
                        
                        # Clean up chromatogram ID for use as feature name
                        clean_id = chrom_id.replace(' ', '_').replace('=', '_').replace('/', '_')
                        clean_id = clean_id.replace('-', '_').replace('.', '_')
                        
                        chromatogram_data.append({
                            'feature_id': clean_id,
                            'original_id': chrom_id,
                            'auc': auc,
                            'n_points': len(intensities),
                            'max_intensity': max(intensities) if len(intensities) > 0 else 0.0,
                            'mean_intensity': sum(intensities) / len(intensities) if len(intensities) > 0 else 0.0
                        })
                    
                except Exception as e:
                    # Skip problematic chromatograms
                    print(f"    Warning: Could not process chromatogram {chrom_id}: {e}")
                    continue
            
            print(f"  Successfully extracted {len(chromatogram_data)} chromatograms")
            
            if len(chromatogram_data) == 0:
                raise FileTypeError("Found chromatograms but could not extract data from any")
            
            # Create feature names and values from chromatogram data
            feature_names = [f"chrom_{chrom['feature_id']}" for chrom in chromatogram_data]
            feature_values = [chrom['auc'] for chrom in chromatogram_data]
            
            # Metadata for chromatogram data
            metadata = {
                'file_type': 'mzml_chromatogram',
                'source_file': str(file_path),
                'n_samples': 1,
                'n_features': len(feature_names),
                'n_chromatograms': len(chromatogram_data),
                'total_auc': sum(feature_values),
                'max_auc': max(feature_values) if feature_values else 0,
                'mean_auc': sum(feature_values) / len(feature_values) if feature_values else 0,
                'data_type': 'SRM/MRM_chromatograms',
                'chromatogram_count_reported': chromatogram_count,
                'chromatogram_ids_found': len(offset_dict)
            }
        
        else:
            # No chromatograms found, try MS1 spectra (original approach)
            print(f"  No chromatograms found (count={chromatogram_count}, offset_dict={len(offset_dict)}), trying MS1 spectra...")
            
            # Collect all MS1 peaks
            all_peaks = []
            spectrum_count = 0
            
            for spectrum in run:
                if spectrum.ms_level == 1:
                    spectrum_count += 1
                    
                    # Get peaks
                    peaks = spectrum.peaks("raw")
                    if len(peaks) > 0:
                        # Add RT information to each peak
                        rt = spectrum.scan_time_in_minutes() or 0
                        for mz, intensity in peaks:
                            all_peaks.append({
                                'mz': mz,
                                'intensity': intensity,
                                'rt': rt,
                                'spectrum_id': spectrum_count
                            })
                    
                    # Limit for demo purposes
                    if spectrum_count > 100:
                        print(f"  Limiting to first {spectrum_count} spectra for demo")
                        break
            
            print(f"  Extracted {len(all_peaks)} peaks from {spectrum_count} MS1 spectra")
            
            if len(all_peaks) == 0:
                raise FileTypeError("No MS1 peaks or chromatograms found in mzML file")
            
            # Create feature matrix using simple m/z binning
            peaks_df = pd.DataFrame(all_peaks)
            
            # Bin m/z values (0.01 Da bins for demo)
            bin_width = 0.01
            peaks_df['mz_bin'] = (peaks_df['mz'] / bin_width).round() * bin_width
            
            # Aggregate intensities by m/z bin (sum intensities)
            feature_matrix = peaks_df.groupby('mz_bin')['intensity'].sum().reset_index()
            feature_matrix.columns = ['mz', 'intensity']
            
            # Create feature names and values
            feature_names = [f"mz_{mz:.3f}" for mz in feature_matrix['mz']]
            feature_values = feature_matrix['intensity'].values
            
            # Metadata for MS1 data
            metadata = {
                'file_type': 'mzml_ms1',
                'source_file': str(file_path),
                'n_samples': 1,
                'n_features': len(feature_names),
                'n_ms1_spectra': spectrum_count,
                'n_peaks_extracted': len(all_peaks),
                'mz_range': [float(peaks_df['mz'].min()), float(peaks_df['mz'].max())],
                'rt_range': [float(peaks_df['rt'].min()), float(peaks_df['rt'].max())],
                'bin_width': bin_width
            }
        
        # Create a single-sample DataFrame (sample name from filename)
        sample_name = Path(file_path).stem
        
        # Create DataFrame with features as columns
        df = pd.DataFrame([feature_values], 
                         columns=feature_names, 
                         index=[sample_name])
        
        print(f"  Created feature matrix: {df.shape[0]} samples × {df.shape[1]} features")
        return df, metadata
        
    except Exception as e:
        raise FileTypeError(f"Error processing mzML file {file_path}: {e}")


def load_mwtab_file(file_path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Load mwTab file using existing biomarker_metabo logic
    
    Returns:
        Tuple of (metabolite_data_frame, metadata_dict)
    """
    if not MWTAB_AVAILABLE:
        raise FileTypeError("mwtab package not available. Please install with: pip install mwtab")
    
    try:
        # Use existing mwtab loading logic
        mw = next(mwtab.read_files(str(file_path)))
        ms = mw['MS_METABOLITE_DATA']
        
        # Extract data block
        data_block = ms['Data']
        
        if isinstance(data_block, dict):
            df = pd.DataFrame.from_dict(data_block, orient='index')
        elif isinstance(data_block, list):
            df_feat = pd.DataFrame(data_block).set_index('Metabolite')
            df = df_feat.T
        else:
            raise ValueError(f"Unexpected data block type: {type(data_block)}")
        
        # Force numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Extract metabolite names from METABOLITES section if available
        metabolite_names = None
        if 'METABOLITES' in mw:
            try:
                metabolites_data = mw['METABOLITES']
                if isinstance(metabolites_data, list) and len(metabolites_data) > 0:
                    # Extract metabolite names
                    metabolite_names = []
                    for metabolite_entry in metabolites_data:
                        if isinstance(metabolite_entry, dict) and 'metabolite_name' in metabolite_entry:
                            metabolite_names.append(metabolite_entry['metabolite_name'])
                    
                    # Map metabolite names to columns if we have the right number
                    if len(metabolite_names) == len(df.columns):
                        print(f"✅ Found {len(metabolite_names)} metabolite names from METABOLITES section")
                        # Create mapping from old column names to metabolite names
                        column_mapping = dict(zip(df.columns, metabolite_names))
                        df = df.rename(columns=column_mapping)
                    else:
                        print(f"⚠️ Metabolite names count ({len(metabolite_names)}) doesn't match columns ({len(df.columns)})")
                        metabolite_names = None
            except Exception as e:
                print(f"⚠️ Could not extract metabolite names: {e}")
                metabolite_names = None
        
        # Extract metadata
        metadata = {
            'file_type': 'mwtab',
            'source_file': str(file_path),
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'study_id': mw.get('STUDY', {}).get('STUDY_ID', 'unknown'),
            'analysis_id': mw.get('ANALYSIS', {}).get('ANALYSIS_ID', 'unknown'),
            'has_metabolite_names': metabolite_names is not None,
            'metabolite_names': metabolite_names
        }
        
        return df, metadata
        
    except Exception as e:
        raise FileTypeError(f"Error loading mwTab file {file_path}: {e}")


def load_file(file_path: Union[str, Path], file_type: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Load file based on detected or specified type
    
    Args:
        file_path: Path to input file  
        file_type: Optional explicit file type (will auto-detect if None)
        
    Returns:
        Tuple of (data_frame, metadata_dict)
        
    Raises:
        FileTypeError: If file cannot be loaded
    """
    if file_type is None:
        file_type = detect_file_type(file_path)
    
    print(f"Loading file {file_path} as type: {file_type}")
    
    if file_type == 'mwtab':
        return load_mwtab_file(file_path)
    elif file_type in ['mzml_centroid', 'mzml_profile']:
        return extract_ms1_data_from_mzml(file_path)
    else:
        raise FileTypeError(f"Unsupported file type: {file_type}")


# CLI interface
def main():
    """Command line interface for file loader"""
    if len(sys.argv) < 2:
        print("Usage: python file_loader.py <input_file> [--type=mwtab|mzml_centroid|mzml_profile]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    file_type = None
    
    # Parse optional type argument
    for arg in sys.argv[2:]:
        if arg.startswith('--type='):
            file_type = arg.split('=', 1)[1]
    
    try:
        df, metadata = load_file(input_file, file_type)
        print(f"\nSuccessfully loaded file:")
        print(f"  File type: {metadata['file_type']}")
        print(f"  Data shape: {df.shape}")
        
        # Print specific metadata based on file type
        if metadata['file_type'] == 'mwtab':
            print(f"  Study ID: {metadata.get('study_id', 'N/A')}")
            print(f"  Analysis ID: {metadata.get('analysis_id', 'N/A')}")
        elif 'mzml' in metadata['file_type']:
            print(f"  MS1 spectra: {metadata.get('n_ms1_spectra', 'N/A')}")
            print(f"  m/z range: {metadata.get('mz_range', 'N/A')}")
            print(f"  RT range: {metadata.get('rt_range', 'N/A')} min")
        
        print(f"\nFirst few samples: {list(df.index[:5])}")
        print(f"First few features: {list(df.columns[:5])}")
        
    except FileTypeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 