#!/usr/bin/env python3
"""
Full Dataset Processing Test
Process all raw mzML files with real labels and compare with mwTab results
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import time

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

def extract_real_labels_from_excel(excel_file: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Extract real sample labels from the Excel mapping file
    
    Returns:
        Tuple of (mapping_dataframe, sample_id_to_label_dict)
    """
    try:
        print(f"Extracting real labels from {excel_file}")
        df = pd.read_excel(excel_file)
        
        # The file has headers in the first row, so we need to use row 1 as column names
        df.columns = df.iloc[0]  # Use first row as column names
        df = df.drop(0).reset_index(drop=True)  # Drop the header row
        
        # Extract sample IDs and group labels
        sample_ids = df['Sample'].astype(int)
        group_labels = df['GroupName']
        
        # Create mapping dictionary
        sample_to_label = dict(zip(sample_ids, group_labels))
        
        print(f"  Extracted labels for {len(sample_to_label)} samples")
        
        # Count labels
        label_counts = group_labels.value_counts()
        print(f"  Label distribution: {label_counts.to_dict()}")
        
        return df, sample_to_label
        
    except Exception as e:
        print(f"Error extracting labels from Excel: {e}")
        return None, {}


def find_all_mzml_files(imss_dir: str) -> List[Path]:
    """Find all mzML files in IMSS directory"""
    imss_path = Path(imss_dir)
    if not imss_path.exists():
        raise FileNotFoundError(f"IMSS directory not found: {imss_dir}")
    
    mzml_files = list(imss_path.glob("*.mzML"))
    if len(mzml_files) == 0:
        raise FileNotFoundError(f"No mzML files found in {imss_dir}")
    
    # Sort files for consistent processing
    mzml_files = sorted(mzml_files)
    print(f"Found {len(mzml_files)} mzML files")
    
    return mzml_files


def extract_sample_id_from_filename(filename: str) -> int:
    """
    Extract sample ID from mzML filename
    Expected format: "Lipidyzer Batch - 20181024111314 - 1-29664 - 01.mzML"
    The sample ID is the last number before .mzML
    """
    try:
        # Extract the last number before .mzML
        basename = Path(filename).stem
        # Split by '-' and get the last part, then convert to int
        parts = basename.split('-')
        sample_id = int(parts[-1].strip())
        return sample_id
    except:
        # Fallback: try to extract any number from filename
        import re
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[-1])
        else:
            return -1  # Invalid


def process_all_mzml_files(mzml_files: List[Path], sample_to_label: Dict[int, str], max_files: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Process all mzML files and extract real labels
    
    Args:
        mzml_files: List of mzML file paths
        sample_to_label: Dictionary mapping sample IDs to labels
        max_files: Optional limit for testing (None = process all)
    
    Returns:
        Tuple of (feature_matrix, labels_series)
    """
    from biomarker.ingest.file_loader import load_file
    
    if max_files:
        mzml_files = mzml_files[:max_files]
        print(f"Processing {len(mzml_files)} mzML files (limited for testing)")
    else:
        print(f"Processing ALL {len(mzml_files)} mzML files")
    
    print(f"\n--- Processing mzML Files ---")
    
    all_samples = []
    all_metadata = []
    sample_labels = []
    processed_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, mzml_file in enumerate(mzml_files):
        try:
            if i % 50 == 0:  # Progress update every 50 files
                elapsed = time.time() - start_time
                print(f"  Progress: {i}/{len(mzml_files)} files ({i/len(mzml_files)*100:.1f}%) - {elapsed:.1f}s elapsed")
            
            # Extract sample ID from filename
            sample_id = extract_sample_id_from_filename(mzml_file.name)
            
            # Check if we have a label for this sample
            if sample_id not in sample_to_label:
                print(f"    ⚠️ No label found for sample {sample_id} ({mzml_file.name}) - skipping")
                failed_count += 1
                continue
            
            # Load single mzML file
            df_single, metadata = load_file(str(mzml_file))
            
            # Add to collection
            all_samples.append(df_single)
            all_metadata.append(metadata)
            sample_labels.append(sample_to_label[sample_id])
            
            processed_count += 1
            
            if i < 5:  # Show details for first few files
                print(f"    ✓ {mzml_file.name} -> Sample {sample_id} -> {sample_to_label[sample_id]} ({df_single.shape[1]} features)")
            
        except Exception as e:
            print(f"    ❌ Error processing {mzml_file.name}: {e}")
            failed_count += 1
            continue
    
    total_time = time.time() - start_time
    print(f"\n--- Processing Complete ---")
    print(f"  Successfully processed: {processed_count} files")
    print(f"  Failed: {failed_count} files")
    print(f"  Total time: {total_time:.1f}s ({total_time/len(mzml_files):.2f}s per file)")
    
    if len(all_samples) == 0:
        raise ValueError("No mzML files were successfully processed")
    
    # Combine all samples into one DataFrame
    print(f"\n--- Combining {len(all_samples)} samples ---")
    
    # Get all unique features across samples
    all_features = set()
    for df in all_samples:
        all_features.update(df.columns)
    
    all_features = sorted(list(all_features))
    print(f"Total unique features across all samples: {len(all_features)}")
    
    # Create combined DataFrame
    combined_data = []
    sample_names = []
    
    for df in all_samples:
        # Align features (fill missing with 0)
        aligned_sample = []
        for feature in all_features:
            if feature in df.columns:
                aligned_sample.append(df[feature].iloc[0])
            else:
                aligned_sample.append(0.0)
        
        combined_data.append(aligned_sample)
        sample_names.append(df.index[0])
    
    # Create final DataFrame
    combined_df = pd.DataFrame(
        combined_data, 
        columns=all_features, 
        index=sample_names
    )
    
    # Create labels series
    labels_series = pd.Series(sample_labels, index=sample_names, name='Group')
    
    print(f"✓ Combined feature matrix: {combined_df.shape}")
    print(f"✓ Label distribution: {labels_series.value_counts().to_dict()}")
    
    return combined_df, labels_series


def run_complete_biomarker_analysis(df: pd.DataFrame, labels: pd.Series, output_dir: str):
    """Run the complete biomarker discovery pipeline"""
    from biomarker.qc.qc_plots import generate_qc_summary_plot
    from biomarker.ml.statistical_analysis_simple import run_complete_statistical_analysis
    from biomarker.ml.model_pipeline import train_logistic_regression
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Running Complete Biomarker Analysis ---")
    
    # Step 1: QC Plots
    print("Step 1: Generating QC plots...")
    try:
        qc_plot = generate_qc_summary_plot(df, str(output_path / "qc_summary.png"))
        print(f"  ✓ QC plots saved: {qc_plot}")
    except Exception as e:
        print(f"  ⚠️ QC plots failed: {e}")
    
    # Step 2: Statistical Analysis
    print("Step 2: Statistical analysis...")
    try:
        stat_results = run_complete_statistical_analysis(
            df, labels, str(output_path / "statistical"), alpha=0.05
        )
        
        print(f"  ✓ Features tested: {stat_results['summary']['n_features_tested']}")
        print(f"  ✓ Significant (raw p<0.05): {stat_results['summary']['n_significant_raw']}")
        print(f"  ✓ Significant (adj p<0.05): {stat_results['summary']['n_significant_adj']}")
        
        # Show top features
        top_features = stat_results['summary']['top_features'][:10]
        print("  Top 10 features from raw mzML:")
        for i, feat in enumerate(top_features):
            print(f"    {i+1}. {feat['feature']}: p={feat['p_value']:.2e}, FC={feat['fold_change']:.2f}")
            
    except Exception as e:
        print(f"  ❌ Statistical analysis failed: {e}")
        stat_results = None
    
    # Step 3: ML Training
    print("Step 3: ML model training...")
    try:
        ml_results = train_logistic_regression(
            df, labels, str(output_path / "ml_model")
        )
        
        print(f"  ✓ Cross-validation AUC: {ml_results['cv_auc_mean']:.3f} ± {ml_results['cv_auc_std']:.3f}")
        print(f"  ✓ Final test AUC: {ml_results['final_auc']:.3f}")
        print(f"  ✓ Model artifacts saved in: {output_path / 'ml_model'}")
        
    except Exception as e:
        print(f"  ❌ ML training failed: {e}")
        ml_results = None
    
    return {
        'statistical_results': stat_results,
        'ml_results': ml_results
    }


def compare_mwtab_vs_raw_results(raw_stat_file: str, raw_ml_file: str):
    """Compare raw mzML results with mwTab results"""
    print(f"\n--- Comparing mwTab vs Raw mzML Results ---")
    
    # mwTab results files
    mwtab_stat_file = "test_statistical_output/statistical_results.csv"
    mwtab_ml_summary = "test_model_output/model_summary.json"
    
    # Compare statistical results
    if os.path.exists(mwtab_stat_file) and os.path.exists(raw_stat_file):
        print("📊 Statistical Results Comparison:")
        
        mwtab_stats = pd.read_csv(mwtab_stat_file)
        raw_stats = pd.read_csv(raw_stat_file)
        
        print(f"  mwTab: {mwtab_stats['significant'].sum()} significant features out of {len(mwtab_stats)}")
        print(f"  Raw mzML: {raw_stats['significant'].sum()} significant features out of {len(raw_stats)}")
        
        # Show top features from each
        print("\n  Top 5 mwTab features:")
        for i, row in mwtab_stats.head(5).iterrows():
            print(f"    {i+1}. {row['feature']}: p={row['p_value']:.2e}")
        
        print("\n  Top 5 Raw mzML features:")
        for i, row in raw_stats.head(5).iterrows():
            print(f"    {i+1}. {row['feature']}: p={row['p_value']:.2e}")
    
    # Compare ML results
    if os.path.exists(mwtab_ml_summary) and os.path.exists(raw_ml_file):
        print("\n🤖 ML Results Comparison:")
        
        import json
        with open(mwtab_ml_summary, 'r') as f:
            mwtab_ml = json.load(f)
        with open(raw_ml_file, 'r') as f:
            raw_ml = json.load(f)
        
        print(f"  mwTab CV AUC: {mwtab_ml.get('cv_auc_mean', 'N/A'):.3f} ± {mwtab_ml.get('cv_auc_std', 0):.3f}")
        print(f"  Raw mzML CV AUC: {raw_ml.get('cv_auc_mean', 'N/A'):.3f} ± {raw_ml.get('cv_auc_std', 0):.3f}")
        print(f"  mwTab Final AUC: {mwtab_ml.get('final_auc', 'N/A'):.3f}")
        print(f"  Raw mzML Final AUC: {raw_ml.get('final_auc', 'N/A'):.3f}")


def main():
    """Main function for full dataset processing"""
    print("=" * 80)
    print("FULL DATASET PROCESSING - mwTab vs Raw mzML Comparison")
    print("=" * 80)
    
    # Configuration
    imss_dir = "./IMSS"
    mapping_file = "./IMSS/Map_IMSS.xlsx"
    output_dir = "test_full_dataset_output"
    
    # For testing, set max_files to limit processing
    # Set to None to process ALL files
    max_files = None  # Process ALL 400 files!
    
    # Clean up previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        total_start_time = time.time()
        
        # Step 1: Extract real labels
        print("Step 1: Extracting real labels from Excel mapping...")
        mapping_df, sample_to_label = extract_real_labels_from_excel(mapping_file)
        
        if not sample_to_label:
            raise ValueError("Failed to extract labels from Excel file")
        
        # Step 2: Find all mzML files
        print("\nStep 2: Finding all mzML files...")
        mzml_files = find_all_mzml_files(imss_dir)
        
        # Step 3: Process all mzML files with real labels
        print("\nStep 3: Processing mzML files with real labels...")
        feature_matrix, labels = process_all_mzml_files(mzml_files, sample_to_label, max_files)
        
        # Step 4: Run complete biomarker analysis
        print("\nStep 4: Running complete biomarker analysis...")
        results = run_complete_biomarker_analysis(feature_matrix, labels, output_dir)
        
        # Step 5: Compare with mwTab results
        print("\nStep 5: Comparing with mwTab results...")
        raw_stat_file = f"{output_dir}/statistical/statistical_results.csv"
        raw_ml_file = f"{output_dir}/ml_model/model_summary.json"
        compare_mwtab_vs_raw_results(raw_stat_file, raw_ml_file)
        
        # Final summary
        total_time = time.time() - total_start_time
        print("\n" + "=" * 80)
        print("FULL DATASET PROCESSING - SUMMARY")
        print("=" * 80)
        print(f"✅ Successfully processed {feature_matrix.shape[0]} raw mzML samples")
        print(f"✅ Feature matrix: {feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features")
        print(f"✅ Real labels extracted: {labels.value_counts().to_dict()}")
        print(f"✅ Total processing time: {total_time/60:.1f} minutes")
        print(f"✅ Results saved in: {output_dir}")
        
        if results['statistical_results']:
            stat_summary = results['statistical_results']['summary']
            print(f"✅ Significant biomarkers found: {stat_summary['n_significant_adj']} (adj p<0.05)")
        
        if results['ml_results']:
            ml_summary = results['ml_results']
            print(f"✅ ML model performance: AUC = {ml_summary['final_auc']:.3f}")
        
        print("\n🎉 Full dataset processing completed successfully!")
        print("🔬 Ready to compare biomarker discoveries between mwTab and raw mzML!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Full dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    
    if success:
        print("\n✅ Full dataset test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Full dataset test FAILED!")
        sys.exit(1) 