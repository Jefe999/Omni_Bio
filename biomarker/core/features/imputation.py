#!/usr/bin/env python3
"""
Task #8: Missing Value Imputation
Median-per-cohort and other imputation strategies.

Target: Imputed matrix has no NaNs and preserves biological variance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class ImputationError(Exception):
    """Raised when imputation fails"""
    pass


def impute_missing_values(
    feature_df: pd.DataFrame,
    method: str = 'median_per_cohort',
    group_column: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values in feature matrix
    
    Args:
        feature_df: Feature matrix (samples × features)
        method: Imputation method ('median_per_cohort', 'median_global', 'mean_per_cohort', 
                'mean_global', 'knn', 'iterative', 'zero', 'min_value')
        group_column: Column name for sample groups (required for per-cohort methods)
        **kwargs: Additional parameters for specific imputation methods
        
    Returns:
        Tuple of (imputed_df, imputation_stats)
    """
    print(f"🔧 Starting missing value imputation ({method})...")
    
    # Check for missing values
    initial_missing = feature_df.isna().sum().sum()
    total_values = feature_df.size
    missing_percentage = (initial_missing / total_values) * 100
    
    print(f"  Missing values: {initial_missing:,} / {total_values:,} ({missing_percentage:.2f}%)")
    
    if initial_missing == 0:
        print("  ✅ No missing values found - returning original data")
        stats = {
            'method': method,
            'initial_missing': 0,
            'final_missing': 0,
            'values_imputed': 0,
            'imputation_rate': 0.0
        }
        return feature_df.copy(), stats
    
    # Apply imputation method
    if method == 'median_per_cohort':
        imputed_df, method_stats = impute_median_per_cohort(feature_df, group_column)
    elif method == 'median_global':
        imputed_df, method_stats = impute_median_global(feature_df)
    elif method == 'mean_per_cohort':
        imputed_df, method_stats = impute_mean_per_cohort(feature_df, group_column)
    elif method == 'mean_global':
        imputed_df, method_stats = impute_mean_global(feature_df)
    elif method == 'knn':
        imputed_df, method_stats = impute_knn(feature_df, **kwargs)
    elif method == 'iterative':
        imputed_df, method_stats = impute_iterative(feature_df, **kwargs)
    elif method == 'zero':
        imputed_df, method_stats = impute_zero(feature_df)
    elif method == 'min_value':
        imputed_df, method_stats = impute_min_value(feature_df, **kwargs)
    else:
        raise ImputationError(f"Unknown imputation method: {method}")
    
    # Verify no missing values remain
    final_missing = imputed_df.isna().sum().sum()
    
    # Compile statistics
    stats = {
        'method': method,
        'initial_missing': int(initial_missing),
        'final_missing': int(final_missing),
        'values_imputed': int(initial_missing - final_missing),
        'imputation_rate': ((initial_missing - final_missing) / initial_missing) * 100 if initial_missing > 0 else 0,
        'missing_percentage_before': missing_percentage,
        'missing_percentage_after': (final_missing / total_values) * 100,
        'method_specific_stats': method_stats
    }
    
    # Preserve metadata if present
    if 'feature_metadata' in feature_df.attrs:
        imputed_df.attrs['feature_metadata'] = feature_df.attrs['feature_metadata']
    
    print(f"  ✅ Imputation complete!")
    print(f"    Values imputed: {stats['values_imputed']:,}")
    print(f"    Remaining missing: {final_missing}")
    print(f"    Success rate: {stats['imputation_rate']:.1f}%")
    
    return imputed_df, stats


def impute_median_per_cohort(
    feature_df: pd.DataFrame,
    group_column: Optional[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using median per cohort/group
    
    Args:
        feature_df: Feature matrix
        group_column: Column containing group information
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    if group_column is None:
        # Fall back to global median
        return impute_median_global(feature_df)
    
    # Separate group information from features
    if group_column in feature_df.columns:
        groups = feature_df[group_column]
        feature_data = feature_df.drop(columns=[group_column])
    else:
        # Assume group information is in index or provided separately
        raise ImputationError(f"Group column '{group_column}' not found in DataFrame")
    
    imputed_data = feature_data.copy()
    group_stats = {}
    
    for group_name in groups.unique():
        group_mask = groups == group_name
        group_data = feature_data.loc[group_mask]
        
        # Calculate median for each feature in this group
        group_medians = group_data.median(axis=0, skipna=True)
        
        # Impute missing values with group medians
        for feature in feature_data.columns:
            missing_mask = group_mask & feature_data[feature].isna()
            if missing_mask.any():
                imputed_data.loc[missing_mask, feature] = group_medians[feature]
        
        # Track statistics per group
        group_missing_before = group_data.isna().sum().sum()
        group_missing_after = imputed_data.loc[group_mask].isna().sum().sum()
        
        group_stats[str(group_name)] = {
            'samples': int(group_mask.sum()),
            'missing_before': int(group_missing_before),
            'missing_after': int(group_missing_after),
            'imputed': int(group_missing_before - group_missing_after)
        }
    
    # Handle any remaining missing values with global median
    remaining_missing = imputed_data.isna().sum().sum()
    if remaining_missing > 0:
        global_medians = imputed_data.median(axis=0, skipna=True)
        for feature in imputed_data.columns:
            missing_mask = imputed_data[feature].isna()
            if missing_mask.any():
                imputed_data.loc[missing_mask, feature] = global_medians[feature]
    
    # Reconstruct DataFrame with group column
    if group_column in feature_df.columns:
        imputed_df = imputed_data.copy()
        imputed_df[group_column] = groups
    else:
        imputed_df = imputed_data
    
    method_stats = {
        'groups_processed': len(groups.unique()),
        'group_statistics': group_stats,
        'fallback_global_imputation': remaining_missing > 0
    }
    
    return imputed_df, method_stats


def impute_median_global(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using global median for each feature
    
    Args:
        feature_df: Feature matrix
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    imputed_df = feature_df.copy()
    
    # Calculate global medians for each feature
    feature_medians = imputed_df.median(axis=0, skipna=True)
    
    # Impute missing values
    imputation_counts = {}
    for feature in imputed_df.columns:
        missing_mask = imputed_df[feature].isna()
        if missing_mask.any():
            imputed_df.loc[missing_mask, feature] = feature_medians[feature]
            imputation_counts[feature] = int(missing_mask.sum())
    
    method_stats = {
        'features_imputed': len(imputation_counts),
        'imputation_counts_per_feature': imputation_counts,
        'median_values_used': feature_medians.to_dict()
    }
    
    return imputed_df, method_stats


def impute_mean_per_cohort(
    feature_df: pd.DataFrame,
    group_column: Optional[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using mean per cohort/group
    
    Args:
        feature_df: Feature matrix
        group_column: Column containing group information
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    if group_column is None:
        return impute_mean_global(feature_df)
    
    # Similar logic to median_per_cohort but using mean
    if group_column in feature_df.columns:
        groups = feature_df[group_column]
        feature_data = feature_df.drop(columns=[group_column])
    else:
        raise ImputationError(f"Group column '{group_column}' not found in DataFrame")
    
    imputed_data = feature_data.copy()
    group_stats = {}
    
    for group_name in groups.unique():
        group_mask = groups == group_name
        group_data = feature_data.loc[group_mask]
        
        # Calculate mean for each feature in this group
        group_means = group_data.mean(axis=0, skipna=True)
        
        # Impute missing values with group means
        for feature in feature_data.columns:
            missing_mask = group_mask & feature_data[feature].isna()
            if missing_mask.any():
                imputed_data.loc[missing_mask, feature] = group_means[feature]
        
        group_missing_before = group_data.isna().sum().sum()
        group_missing_after = imputed_data.loc[group_mask].isna().sum().sum()
        
        group_stats[str(group_name)] = {
            'samples': int(group_mask.sum()),
            'missing_before': int(group_missing_before),
            'missing_after': int(group_missing_after),
            'imputed': int(group_missing_before - group_missing_after)
        }
    
    # Handle remaining missing values with global mean
    remaining_missing = imputed_data.isna().sum().sum()
    if remaining_missing > 0:
        global_means = imputed_data.mean(axis=0, skipna=True)
        for feature in imputed_data.columns:
            missing_mask = imputed_data[feature].isna()
            if missing_mask.any():
                imputed_data.loc[missing_mask, feature] = global_means[feature]
    
    # Reconstruct DataFrame
    if group_column in feature_df.columns:
        imputed_df = imputed_data.copy()
        imputed_df[group_column] = groups
    else:
        imputed_df = imputed_data
    
    method_stats = {
        'groups_processed': len(groups.unique()),
        'group_statistics': group_stats,
        'fallback_global_imputation': remaining_missing > 0
    }
    
    return imputed_df, method_stats


def impute_mean_global(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using global mean for each feature
    
    Args:
        feature_df: Feature matrix
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    imputed_df = feature_df.copy()
    
    # Calculate global means for each feature
    feature_means = imputed_df.mean(axis=0, skipna=True)
    
    # Impute missing values
    imputation_counts = {}
    for feature in imputed_df.columns:
        missing_mask = imputed_df[feature].isna()
        if missing_mask.any():
            imputed_df.loc[missing_mask, feature] = feature_means[feature]
            imputation_counts[feature] = int(missing_mask.sum())
    
    method_stats = {
        'features_imputed': len(imputation_counts),
        'imputation_counts_per_feature': imputation_counts,
        'mean_values_used': feature_means.to_dict()
    }
    
    return imputed_df, method_stats


def impute_knn(
    feature_df: pd.DataFrame,
    n_neighbors: int = 5,
    weights: str = 'uniform'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using K-Nearest Neighbors
    
    Args:
        feature_df: Feature matrix
        n_neighbors: Number of neighbors to use
        weights: Weight function for neighbors
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    # Separate numeric columns for imputation
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = feature_df.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_columns) == 0:
        raise ImputationError("No numeric columns found for KNN imputation")
    
    # Apply KNN imputation to numeric columns
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_numeric = imputer.fit_transform(feature_df[numeric_columns])
    
    # Create imputed DataFrame
    imputed_df = feature_df.copy()
    imputed_df[numeric_columns] = imputed_numeric
    
    method_stats = {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'numeric_columns_imputed': len(numeric_columns),
        'non_numeric_columns_preserved': len(non_numeric_columns)
    }
    
    return imputed_df, method_stats


def impute_iterative(
    feature_df: pd.DataFrame,
    max_iter: int = 10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values using Iterative Imputer (MICE-like)
    
    Args:
        feature_df: Feature matrix
        max_iter: Maximum number of imputation rounds
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    # Separate numeric columns for imputation
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        raise ImputationError("No numeric columns found for iterative imputation")
    
    # Apply iterative imputation
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_numeric = imputer.fit_transform(feature_df[numeric_columns])
    
    # Create imputed DataFrame
    imputed_df = feature_df.copy()
    imputed_df[numeric_columns] = imputed_numeric
    
    method_stats = {
        'max_iter': max_iter,
        'random_state': random_state,
        'converged': imputer.n_iter_ < max_iter,
        'iterations_performed': int(imputer.n_iter_),
        'numeric_columns_imputed': len(numeric_columns)
    }
    
    return imputed_df, method_stats


def impute_zero(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values with zeros
    
    Args:
        feature_df: Feature matrix
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    imputed_df = feature_df.fillna(0)
    
    method_stats = {
        'imputation_value': 0,
        'total_values_imputed': (feature_df.isna().sum().sum())
    }
    
    return imputed_df, method_stats


def impute_min_value(
    feature_df: pd.DataFrame,
    fraction: float = 0.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values with a fraction of the minimum value per feature
    
    Args:
        feature_df: Feature matrix
        fraction: Fraction of minimum value to use for imputation
        
    Returns:
        Tuple of (imputed_df, method_stats)
    """
    imputed_df = feature_df.copy()
    
    # Calculate minimum values per feature
    feature_mins = feature_df.min(axis=0, skipna=True)
    imputation_values = feature_mins * fraction
    
    # Impute missing values
    for feature in imputed_df.columns:
        missing_mask = imputed_df[feature].isna()
        if missing_mask.any():
            imputed_df.loc[missing_mask, feature] = imputation_values[feature]
    
    method_stats = {
        'fraction_of_min': fraction,
        'imputation_values_per_feature': imputation_values.to_dict()
    }
    
    return imputed_df, method_stats


def analyze_missing_patterns(feature_df: pd.DataFrame) -> Dict:
    """
    Analyze missing value patterns in the dataset
    
    Args:
        feature_df: Feature matrix
        
    Returns:
        Dictionary with missing value analysis
    """
    total_values = feature_df.size
    missing_values = feature_df.isna()
    
    # Overall statistics
    total_missing = missing_values.sum().sum()
    missing_percentage = (total_missing / total_values) * 100
    
    # Per-feature statistics
    feature_missing = missing_values.sum(axis=0)
    feature_missing_pct = (feature_missing / len(feature_df)) * 100
    
    # Per-sample statistics
    sample_missing = missing_values.sum(axis=1)
    sample_missing_pct = (sample_missing / len(feature_df.columns)) * 100
    
    # Missing patterns
    missing_patterns = missing_values.apply(lambda x: tuple(x), axis=1).value_counts()
    
    analysis = {
        'total_values': int(total_values),
        'total_missing': int(total_missing),
        'missing_percentage': float(missing_percentage),
        'features_with_missing': int((feature_missing > 0).sum()),
        'samples_with_missing': int((sample_missing > 0).sum()),
        'features_completely_missing': int((feature_missing == len(feature_df)).sum()),
        'samples_completely_missing': int((sample_missing == len(feature_df.columns)).sum()),
        'feature_missing_stats': {
            'min': float(feature_missing_pct.min()),
            'max': float(feature_missing_pct.max()),
            'mean': float(feature_missing_pct.mean()),
            'median': float(feature_missing_pct.median())
        },
        'sample_missing_stats': {
            'min': float(sample_missing_pct.min()),
            'max': float(sample_missing_pct.max()),
            'mean': float(sample_missing_pct.mean()),
            'median': float(sample_missing_pct.median())
        },
        'unique_missing_patterns': len(missing_patterns),
        'most_common_patterns': missing_patterns.head(5).to_dict()
    }
    
    return analysis


def save_imputation_report(
    stats: Dict,
    output_file: Path
) -> None:
    """
    Save imputation statistics to JSON file
    
    Args:
        stats: Imputation statistics
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"  📄 Imputation report saved: {output_file}")


# CLI interface
def main():
    """Command line interface for missing value imputation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Missing Value Imputation")
    parser.add_argument("feature_matrix", help="Input feature matrix CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--method", default='median_per_cohort',
                       choices=['median_per_cohort', 'median_global', 'mean_per_cohort', 
                               'mean_global', 'knn', 'iterative', 'zero', 'min_value'],
                       help="Imputation method")
    parser.add_argument("--group-column", help="Column name for sample groups")
    parser.add_argument("--analyze-only", action='store_true', help="Only analyze missing patterns")
    
    # Method-specific parameters
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--iterative-max-iter", type=int, default=10, help="Max iterations for iterative imputer")
    parser.add_argument("--min-value-fraction", type=float, default=0.5, help="Fraction of min value for min_value method")
    
    args = parser.parse_args()
    
    try:
        # Load feature matrix
        print(f"Loading feature matrix from {args.feature_matrix}...")
        feature_df = pd.read_csv(args.feature_matrix, index_col=0)
        
        # Analyze missing patterns
        print("Analyzing missing value patterns...")
        missing_analysis = analyze_missing_patterns(feature_df)
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save missing pattern analysis
        analysis_file = output_dir / "missing_value_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(missing_analysis, f, indent=2, default=str)
        
        print(f"Missing value analysis saved: {analysis_file}")
        print(f"  Total missing: {missing_analysis['total_missing']:,} ({missing_analysis['missing_percentage']:.2f}%)")
        print(f"  Features with missing: {missing_analysis['features_with_missing']}")
        print(f"  Samples with missing: {missing_analysis['samples_with_missing']}")
        
        if args.analyze_only:
            print("Analysis complete (analyze-only mode)")
            return
        
        # Run imputation
        method_kwargs = {}
        if args.method == 'knn':
            method_kwargs['n_neighbors'] = args.knn_neighbors
        elif args.method == 'iterative':
            method_kwargs['max_iter'] = args.iterative_max_iter
        elif args.method == 'min_value':
            method_kwargs['fraction'] = args.min_value_fraction
        
        imputed_df, stats = impute_missing_values(
            feature_df,
            method=args.method,
            group_column=args.group_column,
            **method_kwargs
        )
        
        # Save results
        output_file = output_dir / "imputed_features.csv"
        imputed_df.to_csv(output_file)
        
        report_file = output_dir / "imputation_report.json"
        save_imputation_report(stats, report_file)
        
        print(f"\n✅ Imputation completed!")
        print(f"Output: {output_file}")
        print(f"Report: {report_file}")
        
        # Verify no missing values remain
        final_missing = imputed_df.isna().sum().sum()
        if final_missing == 0:
            print("  ✅ SUCCESS: No missing values remain")
        else:
            print(f"  ⚠️ WARNING: {final_missing} missing values still remain")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 