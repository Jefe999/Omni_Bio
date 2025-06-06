"""
Batch mzML Processing for Biomarker Discovery

This module processes multiple mzML files to create a combined feature matrix
suitable for biomarker analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import glob
from tqdm import tqdm
from datetime import datetime

from .file_loader import extract_ms1_data_from_mzml


def process_mzml_batch(
    mzml_files: List[Union[str, Path]], 
    max_files: Optional[int] = None,
    progress_bar: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Process multiple mzML files and create a combined feature matrix
    
    Args:
        mzml_files: List of paths to mzML files
        max_files: Maximum number of files to process (None for all)
        progress_bar: Whether to show progress bar
        
    Returns:
        Tuple of (feature_matrix_dataframe, metadata_dict)
    """
    print(f"🧪 Processing batch of {len(mzml_files)} mzML files...")
    
    if max_files:
        mzml_files = mzml_files[:max_files]
        print(f"   Limited to first {len(mzml_files)} files")
    
    # Store results for each file
    all_dataframes = []
    all_metadata = []
    
    # Process each file
    iterator = tqdm(mzml_files, desc="Processing mzML files") if progress_bar else mzml_files
    
    for i, mzml_file in enumerate(iterator):
        try:
            mzml_path = Path(mzml_file)
            
            if not mzml_path.exists():
                print(f"⚠️ File not found: {mzml_path}")
                continue
            
            print(f"   Processing {i+1}/{len(mzml_files)}: {mzml_path.name}")
            
            # Extract chromatogram data from this file
            df, metadata = extract_ms1_data_from_mzml(str(mzml_path))
            
            if df is not None and not df.empty:
                all_dataframes.append(df)
                all_metadata.append(metadata)
                print(f"     ✅ Extracted {df.shape[1]} features")
            else:
                print(f"     ⚠️ No data extracted from {mzml_path.name}")
                
        except Exception as e:
            print(f"     ❌ Error processing {mzml_path.name}: {e}")
            continue
    
    if not all_dataframes:
        raise ValueError("No data was successfully extracted from any mzML files")
    
    print(f"\n📊 Successfully processed {len(all_dataframes)} files")
    
    # Combine all feature matrices
    print("🔗 Combining feature matrices...")
    
    # Get all unique feature names across all files
    all_features = set()
    for df in all_dataframes:
        all_features.update(df.columns)
    
    all_features = sorted(list(all_features))
    print(f"   Found {len(all_features)} unique features")
    
    # Create combined matrix with all samples and all features
    combined_data = []
    sample_names = []
    
    for df in all_dataframes:
        for sample_name in df.index:
            sample_names.append(sample_name)
            # Create row with all features, filling missing ones with 0
            row = []
            for feature in all_features:
                if feature in df.columns:
                    row.append(df.loc[sample_name, feature])
                else:
                    row.append(0.0)  # Missing features get 0
            combined_data.append(row)
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(
        combined_data,
        index=sample_names,
        columns=all_features
    )
    
    # Combined metadata
    combined_metadata = {
        'n_files_processed': len(all_dataframes),
        'n_samples': combined_df.shape[0],
        'n_features': combined_df.shape[1],
        'feature_names': all_features,
        'individual_metadata': all_metadata,
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    print(f"✅ Created combined feature matrix: {combined_df.shape[0]} samples × {combined_df.shape[1]} features")
    
    return combined_df, combined_metadata


def create_case_control_labels(
    sample_names: List[str], 
    case_patterns: List[str], 
    control_patterns: List[str]
) -> Dict[str, str]:
    """
    Create case/control labels based on filename patterns
    
    Args:
        sample_names: List of sample names
        case_patterns: Patterns that indicate case samples
        control_patterns: Patterns that indicate control samples
        
    Returns:
        Dict mapping sample names to 'Case' or 'Control'
    """
    labels = {}
    
    for sample_name in sample_names:
        is_case = any(pattern in sample_name for pattern in case_patterns)
        is_control = any(pattern in sample_name for pattern in control_patterns)
        
        if is_case and not is_control:
            labels[sample_name] = 'Case'
        elif is_control and not is_case:
            labels[sample_name] = 'Control'
        else:
            # Try to infer from batch number or other patterns
            if '20181106' in sample_name:
                labels[sample_name] = 'Control'  # Batch 1
            elif '20181107' in sample_name:
                labels[sample_name] = 'Case'     # Batch 2
            else:
                labels[sample_name] = 'Unknown'
    
    return labels


def save_feature_matrix_as_mwtab(feature_matrix, labels, output_file, study_title="mzML Batch Processing"):
    """
    Save feature matrix in mwTab format for biomarker analysis
    
    Args:
        feature_matrix: pandas DataFrame with samples as rows, features as columns
        labels: dict mapping sample names to group labels
        output_file: path to save mwTab file
        study_title: title for the study
    """
    with open(output_file, 'w') as f:
        # mwTab header
        f.write("#METABOLOMICS WORKBENCH\n")
        f.write(f"#STUDY_TITLE:{study_title}\n")
        f.write("#ANALYSIS_TYPE:LIPIDYZER_MS\n")
        f.write("#VERSION             VERSION:1.0\n")
        f.write(f"#CREATED_ON          {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("\n")
        
        # Subject information
        f.write("#SUBJECT\n")
        f.write("#SUBJECT_TYPE:Human\n")
        f.write("#SUBJECT_SPECIES:Homo sapiens\n")
        f.write("\n")
        
        # Sample factor section
        f.write("#SUBJECT_SAMPLE_FACTORS:\n")
        
        # Create cleaner feature names - remove problematic characters and shorten names
        clean_columns = []
        for col in feature_matrix.columns:
            # Replace problematic characters and shorten very long names
            clean_col = col.replace("chrom___SRM_SIC_Q1_", "SRM_Q1_").replace("_sample_30_period_1_experiment_", "_exp").replace("_transition_", "_t")
            clean_col = clean_col.replace("chrom_SRM_SIC_Q1_", "SRM_Q1_").replace("chrom_TIC", "TIC")
            clean_col = clean_col.replace("___", "_").replace("__", "_")
            # Limit length to avoid very long column names
            if len(clean_col) > 50:
                clean_col = clean_col[:47] + "..."
            clean_columns.append(clean_col)
        
        # Header line with cleaner column names
        header_line = "Samples\tFactors\t" + "\t".join(clean_columns)
        f.write(header_line + "\n")
        
        # Data rows
        for sample_name, row in feature_matrix.iterrows():
            group = labels.get(sample_name, "Unknown")
            data_line = f"{sample_name}\tGroup:{group}\t" + "\t".join([str(val) for val in row])
            f.write(data_line + "\n")
    
    print(f"✅ Saved mwTab file with cleaner feature names: {output_file}")
    print(f"   Original columns: {len(feature_matrix.columns)}")
    print(f"   Clean columns: {len(clean_columns)}")
    print(f"   First few clean names: {clean_columns[:5]}") 