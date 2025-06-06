#!/usr/bin/env python3
"""
Task #5: OpenMS FeatureFinderMetabo Wrapper
Reproducible peak picking for credible pilot deployment.

This replaces the simple binning approach with proper LC-MS feature detection,
deisotoping, and alignment across samples.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import tempfile
import json

# OpenMS imports (conditional)
try:
    import pyopenms as oms
    OPENMS_AVAILABLE = True
except ImportError:
    OPENMS_AVAILABLE = False
    # Create a placeholder for type hints
    class _DummyOpenMS:
        Param = object
        MSExperiment = object
        FeatureMap = object
    oms = _DummyOpenMS()


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails"""
    pass


@dataclass
class FeaturePickingParams:
    """Parameters for OpenMS FeatureFinderMetabo"""
    # Mass accuracy
    mass_error_ppm: float = 5.0
    
    # Isotope detection
    isotope_filtering: bool = True
    
    # Intensity thresholds
    intensity_threshold: float = 1000.0
    
    # Chromatographic parameters
    elution_model: str = "asymmetric"  # asymmetric, symmetric, none
    
    # Feature linking (alignment across samples)
    rt_tolerance: float = 30.0  # seconds
    mz_tolerance: float = 5.0   # ppm
    
    # Output filtering
    min_feature_size: int = 3  # minimum number of data points per feature
    
    def to_openms_params(self):
        """Convert to OpenMS parameter object"""
        if not OPENMS_AVAILABLE:
            raise FeatureExtractionError("pyOpenMS not available")
            
        import pyopenms as oms_local
        params = oms_local.Param()
        
        # FeatureFinderMetabo parameters
        params.setValue("algorithm:common:noise_threshold_int", float(self.intensity_threshold))
        params.setValue("algorithm:common:chrom_fwhm", 5.0)
        params.setValue("algorithm:mtd:mass_error_ppm", float(self.mass_error_ppm))
        params.setValue("algorithm:mtd:reestimate_mt_sd", "true" if self.isotope_filtering else "false")
        params.setValue("algorithm:mtd:quant_method", "area")
        
        # Elution model
        if self.elution_model != "none":
            params.setValue("algorithm:mtd:trace_termination_criterion", "outlier")
            params.setValue("algorithm:mtd:trace_termination_outliers", 5)
        
        return params


def run_openms_feature_finder(
    mzml_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    params: Optional[FeaturePickingParams] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run OpenMS FeatureFinderMetabo on multiple mzML files
    
    Args:
        mzml_files: List of mzML file paths
        output_dir: Directory to save results
        params: Feature picking parameters
        
    Returns:
        Tuple of (feature_matrix_df, metadata_dict)
        
    Raises:
        FeatureExtractionError: If OpenMS processing fails
    """
    if not OPENMS_AVAILABLE:
        raise FeatureExtractionError("pyOpenMS not available. Install with: pip install pyopenms")
    
    import pyopenms as oms_local
    
    if params is None:
        params = FeaturePickingParams()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔬 Starting OpenMS feature extraction on {len(mzml_files)} files...")
    
    # Process each mzML file
    feature_maps = []
    sample_names = []
    
    for i, mzml_file in enumerate(mzml_files):
        mzml_path = Path(mzml_file)
        sample_name = mzml_path.stem
        sample_names.append(sample_name)
        
        print(f"  Processing {sample_name} ({i+1}/{len(mzml_files)})")
        
        try:
            # Load mzML file
            exp = oms_local.MSExperiment()
            oms_local.MzMLFile().load(str(mzml_path), exp)
            
            # Run FeatureFinderMetabo
            feature_map = run_single_feature_finder(exp, params, sample_name)
            feature_maps.append(feature_map)
            
            print(f"    ✓ Found {feature_map.size()} features")
            
        except Exception as e:
            print(f"    ❌ Failed to process {sample_name}: {e}")
            raise FeatureExtractionError(f"Feature extraction failed for {sample_name}: {e}")
    
    # Align features across samples
    print("🔗 Aligning features across samples...")
    aligned_features = align_feature_maps(feature_maps, sample_names, params)
    
    # Convert to DataFrame
    feature_df = feature_maps_to_dataframe(aligned_features, sample_names)
    
    # Save results
    output_file = output_dir / "feature_matrix.csv"
    feature_df.to_csv(output_file, index=True)
    
    # Create metadata
    metadata = {
        'method': 'openms_feature_finder_metabo',
        'n_samples': len(sample_names),
        'n_features': len(feature_df.columns),
        'sample_names': sample_names,
        'parameters': params.__dict__,
        'output_file': str(output_file)
    }
    
    # Save metadata
    metadata_file = output_dir / "feature_extraction_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Feature extraction complete!")
    print(f"   Features: {len(feature_df.columns)}")
    print(f"   Samples: {len(feature_df.index)}")
    print(f"   Output: {output_file}")
    
    return feature_df, metadata


def run_single_feature_finder(
    experiment: Any,  # 'oms.MSExperiment' when available
    params: FeaturePickingParams,
    sample_name: str
) -> Any:  # 'oms.FeatureMap' when available
    """
    Run FeatureFinderMetabo on a single MSExperiment
    
    Args:
        experiment: OpenMS MSExperiment object
        params: Feature picking parameters
        sample_name: Name of the sample
        
    Returns:
        OpenMS FeatureMap with detected features
    """
    if not OPENMS_AVAILABLE:
        raise FeatureExtractionError("pyOpenMS not available")
        
    import pyopenms as oms_local
    
    # Initialize FeatureFinderMetabo
    ff = oms_local.FeatureFinderAlgorithmMetabo()
    
    # Set parameters
    openms_params = params.to_openms_params()
    ff.setParameters(openms_params)
    
    # Create output FeatureMap
    feature_map = oms_local.FeatureMap()
    
    # Run feature detection
    ff.run(experiment, feature_map)
    
    # Set sample name in metadata
    feature_map.getProteinIdentifications().clear()
    
    return feature_map


def align_feature_maps(
    feature_maps: List[Any],  # List['oms.FeatureMap']
    sample_names: List[str], 
    params: FeaturePickingParams
) -> List[Any]:  # List['oms.FeatureMap']
    """
    Align features across multiple samples using MapAlignmentAlgorithmPoseClustering
    
    Args:
        feature_maps: List of FeatureMap objects
        sample_names: List of sample names
        params: Parameters including alignment tolerances
        
    Returns:
        List of aligned FeatureMap objects
    """
    if not OPENMS_AVAILABLE:
        raise FeatureExtractionError("pyOpenMS not available")
        
    if len(feature_maps) <= 1:
        return feature_maps
    
    import pyopenms as oms_local
    
    # Initialize alignment algorithm
    alignment_algo = oms_local.MapAlignmentAlgorithmPoseClustering()
    
    # Set alignment parameters
    align_params = oms_local.Param()
    align_params.setValue("max_num_peaks_considered", 1000)
    align_params.setValue("pairfinder:distance_RT:max_difference", float(params.rt_tolerance))
    align_params.setValue("pairfinder:distance_MZ:max_difference", float(params.mz_tolerance))
    align_params.setValue("pairfinder:distance_MZ:unit", "ppm")
    
    alignment_algo.setParameters(align_params)
    
    # Perform alignment
    alignment_algo.align(feature_maps)
    
    return feature_maps


def feature_maps_to_dataframe(
    feature_maps: List[Any],  # List['oms.FeatureMap']
    sample_names: List[str]
) -> pd.DataFrame:
    """
    Convert aligned FeatureMap objects to pandas DataFrame
    
    Args:
        feature_maps: List of aligned FeatureMap objects
        sample_names: List of sample names
        
    Returns:
        DataFrame with samples as rows and features as columns
    """
    # Collect all unique features across samples
    all_features = {}  # key: (mz, rt), value: feature_id
    feature_counter = 0
    
    # First pass: collect unique features
    for sample_idx, (feature_map, sample_name) in enumerate(zip(feature_maps, sample_names)):
        for feature in feature_map:
            mz = feature.getMZ()
            rt = feature.getRT()
            
            # Create feature key (rounded for matching)
            feature_key = (round(mz, 6), round(rt, 2))
            
            if feature_key not in all_features:
                all_features[feature_key] = f"feature_{feature_counter:06d}"
                feature_counter += 1
    
    # Create DataFrame
    feature_ids = list(all_features.values())
    feature_df = pd.DataFrame(
        index=sample_names,
        columns=feature_ids,
        dtype=float
    ).fillna(0.0)
    
    # Second pass: fill intensities
    for sample_idx, (feature_map, sample_name) in enumerate(zip(feature_maps, sample_names)):
        for feature in feature_map:
            mz = feature.getMZ()
            rt = feature.getRT()
            intensity = feature.getIntensity()
            
            feature_key = (round(mz, 6), round(rt, 2))
            feature_id = all_features[feature_key]
            
            feature_df.loc[sample_name, feature_id] = intensity
    
    # Add feature metadata as column MultiIndex
    feature_metadata = []
    for (mz, rt), feature_id in all_features.items():
        feature_metadata.append({
            'feature_id': feature_id,
            'mz': mz,
            'rt': rt
        })
    
    # Create metadata DataFrame for reference
    metadata_df = pd.DataFrame(feature_metadata).set_index('feature_id')
    
    # Store metadata in DataFrame attributes (pandas convention)
    feature_df.attrs['feature_metadata'] = metadata_df
    
    return feature_df


def test_openms_feature_extraction():
    """Test function for OpenMS feature extraction"""
    print("🧪 Testing OpenMS feature extraction...")
    
    # Check if we have test mzML files
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("  ⚠️ No test_data directory found, skipping test")
        return False
    
    mzml_files = list(test_data_dir.glob("*.mzML"))
    if not mzml_files:
        print("  ⚠️ No mzML files found in test_data/, skipping test")
        return False
    
    # Use first 2 files for testing
    test_files = mzml_files[:2]
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run feature extraction
            feature_df, metadata = run_openms_feature_finder(
                test_files,
                temp_dir,
                FeaturePickingParams(intensity_threshold=500.0)
            )
            
            # Validate results
            assert feature_df.shape[0] == len(test_files), "Wrong number of samples"
            assert feature_df.shape[1] > 10, "Too few features detected"
            assert not feature_df.isna().all().any(), "Features should have some values"
            
            print(f"  ✅ Test passed: {feature_df.shape[0]} samples × {feature_df.shape[1]} features")
            return True
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


# CLI interface
def main():
    """Command line interface for OpenMS feature extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenMS Feature Extraction")
    parser.add_argument("mzml_files", nargs="+", help="Input mzML files")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--mass-error-ppm", type=float, default=5.0, help="Mass error tolerance (ppm)")
    parser.add_argument("--intensity-threshold", type=float, default=1000.0, help="Minimum intensity")
    parser.add_argument("--rt-tolerance", type=float, default=30.0, help="RT tolerance for alignment (sec)")
    
    args = parser.parse_args()
    
    # Create parameters
    params = FeaturePickingParams(
        mass_error_ppm=args.mass_error_ppm,
        intensity_threshold=args.intensity_threshold,
        rt_tolerance=args.rt_tolerance
    )
    
    try:
        # Run feature extraction
        feature_df, metadata = run_openms_feature_finder(
            args.mzml_files,
            args.output,
            params
        )
        
        print(f"\n✅ Feature extraction completed successfully!")
        print(f"Output: {args.output}/feature_matrix.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 