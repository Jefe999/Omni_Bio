#!/usr/bin/env python3
"""
Examine the Excel mapping file structure to extract real labels
"""

import pandas as pd
import numpy as np

def examine_mapping_file():
    """Examine the structure of the Excel mapping file"""
    
    try:
        # Load the Excel file
        df = pd.read_excel('IMSS/Map_IMSS.xlsx')
        
        print("=" * 60)
        print("EXCEL MAPPING FILE EXAMINATION")
        print("=" * 60)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        print("\nFirst 15 rows:")
        print(df.head(15))
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nUnique values in each column:")
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"  {col}: {unique_vals[:10]} ({'...' if len(unique_vals) > 10 else ''})")
        
        # Look for potential label columns
        print("\nLooking for potential label/group columns...")
        potential_label_cols = []
        
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) >= 2 and len(unique_vals) <= 10:  # Reasonable number of groups
                print(f"  Potential label column '{col}': {unique_vals}")
                potential_label_cols.append(col)
        
        # Look for sample ID patterns
        print("\nLooking for sample ID patterns...")
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['sample', 'id', 'name', 'file']):
                sample_vals = df[col].dropna().head(10).tolist()
                print(f"  {col}: {sample_vals}")
        
        return df, potential_label_cols
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, []

if __name__ == '__main__':
    df, label_cols = examine_mapping_file()
    
    if df is not None:
        print(f"\n✅ Successfully examined mapping file with {df.shape[0]} rows and {df.shape[1]} columns")
        if label_cols:
            print(f"✅ Found {len(label_cols)} potential label columns: {label_cols}")
        else:
            print("⚠️ No obvious label columns found - may need manual inspection")
    else:
        print("❌ Failed to examine mapping file") 