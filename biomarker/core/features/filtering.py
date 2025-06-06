#!/usr/bin/env python3
"""
Task #7: Frequency & Score Filtering
Apply configurable cutoffs based on cohort frequencies and total scores.

Implements the Excel formula logic for feature scoring and filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
import json


class FilteringError(Exception):
    """Raised when filtering fails"""
    pass


def frequency_filter(
    feature_df: pd.DataFrame,
    min_frequency: float = 0.6,
    group_column: Optional[str] = None,
    per_group: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter features based on frequency of presence across samples
    
    Args:
        feature_df: Feature matrix (samples × features)
        min_frequency: Minimum frequency threshold (0.0-1.0)
        group_column: Column name for sample groups (optional)
        per_group: If True, apply frequency filter per group; if False, across all samples
        
    Returns:
        Tuple of (filtered_df, filter_stats)
    """
    print(f"🔍 Applying frequency filter (≥{min_frequency:.1%})...")
    
    initial_features = len(feature_df.columns)
    
    if group_column and per_group and group_column in feature_df.index.names:
        # Per-group frequency filtering
        filtered_features = apply_per_group_frequency_filter(
            feature_df, min_frequency, group_column
        )
    else:
        # Global frequency filtering
        presence_matrix = feature_df > 0
        feature_frequencies = presence_matrix.sum(axis=0) / len(feature_df)
        filtered_features = feature_frequencies[feature_frequencies >= min_frequency].index
    
    filtered_df = feature_df[filtered_features].copy()
    
    # Preserve metadata if present
    if 'feature_metadata' in feature_df.attrs:
        filtered_metadata = feature_df.attrs['feature_metadata'].loc[filtered_features]
        filtered_df.attrs['feature_metadata'] = filtered_metadata
    
    stats = {
        'initial_features': initial_features,
        'features_after_filter': len(filtered_features),
        'features_removed': initial_features - len(filtered_features),
        'removal_percentage': ((initial_features - len(filtered_features)) / initial_features) * 100,
        'min_frequency_threshold': min_frequency,
        'per_group_filtering': per_group and group_column is not None
    }
    
    print(f"  ✅ Frequency filter complete!")
    print(f"    Remaining: {len(filtered_features)} features")
    print(f"    Removed: {stats['features_removed']} features ({stats['removal_percentage']:.1f}%)")
    
    return filtered_df, stats


def apply_per_group_frequency_filter(
    feature_df: pd.DataFrame,
    min_frequency: float,
    group_column: str
) -> pd.Index:
    """
    Apply frequency filter per group, keeping features present in ≥min_frequency in any group
    
    Args:
        feature_df: Feature matrix with group information
        min_frequency: Minimum frequency threshold
        group_column: Column containing group information
        
    Returns:
        Index of features that pass the filter
    """
    # Get group assignments
    if group_column in feature_df.columns:
        groups = feature_df[group_column]
        feature_data = feature_df.drop(columns=[group_column])
    else:
        raise FilteringError(f"Group column '{group_column}' not found in DataFrame")
    
    # Calculate frequency per group
    valid_features = set()
    
    for group_name in groups.unique():
        group_mask = groups == group_name
        group_data = feature_data.loc[group_mask]
        
        presence_matrix = group_data > 0
        group_frequencies = presence_matrix.sum(axis=0) / len(group_data)
        
        # Features that meet frequency threshold in this group
        valid_in_group = group_frequencies[group_frequencies >= min_frequency].index
        valid_features.update(valid_in_group)
    
    return pd.Index(list(valid_features))


def score_filter(
    feature_df: pd.DataFrame,
    score_method: str = 'total_abundance',
    min_score_percentile: float = 50.0,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter features based on calculated scores
    
    Args:
        feature_df: Feature matrix (samples × features)
        score_method: Scoring method ('total_abundance', 'mean_intensity', 'cv_score', 'composite')
        min_score_percentile: Minimum percentile threshold for scores
        weights: Weights for composite scoring (optional)
        
    Returns:
        Tuple of (filtered_df, filter_stats)
    """
    print(f"📊 Applying score filter ({score_method}, ≥{min_score_percentile}th percentile)...")
    
    initial_features = len(feature_df.columns)
    
    # Calculate scores for all features
    scores = calculate_feature_scores(feature_df, score_method, weights)
    
    # Determine threshold
    score_threshold = np.percentile(scores.values, min_score_percentile)
    
    # Filter features
    passing_features = scores[scores >= score_threshold].index
    filtered_df = feature_df[passing_features].copy()
    
    # Preserve metadata if present
    if 'feature_metadata' in feature_df.attrs:
        filtered_metadata = feature_df.attrs['feature_metadata'].loc[passing_features]
        filtered_df.attrs['feature_metadata'] = filtered_metadata
    
    stats = {
        'initial_features': initial_features,
        'features_after_filter': len(passing_features),
        'features_removed': initial_features - len(passing_features),
        'removal_percentage': ((initial_features - len(passing_features)) / initial_features) * 100,
        'score_method': score_method,
        'min_score_percentile': min_score_percentile,
        'score_threshold': float(score_threshold),
        'score_statistics': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'median': float(scores.median())
        }
    }
    
    print(f"  ✅ Score filter complete!")
    print(f"    Remaining: {len(passing_features)} features")
    print(f"    Removed: {stats['features_removed']} features ({stats['removal_percentage']:.1f}%)")
    print(f"    Score threshold: {score_threshold:.2e}")
    
    return filtered_df, stats


def calculate_feature_scores(
    feature_df: pd.DataFrame,
    method: str,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Calculate scores for all features using specified method
    
    Args:
        feature_df: Feature matrix
        method: Scoring method
        weights: Weights for composite scoring
        
    Returns:
        Series of feature scores
    """
    if method == 'total_abundance':
        return feature_df.sum(axis=0)
    
    elif method == 'mean_intensity':
        # Mean of non-zero values
        return feature_df.apply(lambda col: col[col > 0].mean() if (col > 0).any() else 0, axis=0)
    
    elif method == 'cv_score':
        # Coefficient of variation for non-zero values
        def cv_score(col):
            non_zero = col[col > 0]
            if len(non_zero) < 2:
                return 0
            return non_zero.std() / non_zero.mean() if non_zero.mean() > 0 else 0
        
        return feature_df.apply(cv_score, axis=0)
    
    elif method == 'presence_score':
        # Number of samples with non-zero values
        return (feature_df > 0).sum(axis=0)
    
    elif method == 'composite':
        # Weighted combination of multiple scores
        if weights is None:
            weights = {
                'total_abundance': 0.4,
                'mean_intensity': 0.3,
                'presence_score': 0.2,
                'cv_score': 0.1
            }
        
        composite_scores = pd.Series(0.0, index=feature_df.columns)
        
        for score_type, weight in weights.items():
            if score_type in ['total_abundance', 'mean_intensity', 'cv_score', 'presence_score']:
                scores = calculate_feature_scores(feature_df, score_type)
                # Normalize scores to 0-1 range
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else scores
                composite_scores += weight * normalized_scores
        
        return composite_scores
    
    else:
        raise FilteringError(f"Unknown scoring method: {method}")


def combined_filter(
    feature_df: pd.DataFrame,
    filter_config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply multiple filters in sequence based on configuration
    
    Args:
        feature_df: Feature matrix
        filter_config: Configuration dictionary with filter parameters
        
    Returns:
        Tuple of (filtered_df, comprehensive_stats)
    """
    print("🔧 Applying combined feature filters...")
    
    current_df = feature_df.copy()
    filter_history = []
    
    # Apply frequency filter if configured
    if filter_config.get('frequency_filter', {}).get('enabled', False):
        freq_config = filter_config['frequency_filter']
        current_df, freq_stats = frequency_filter(
            current_df,
            min_frequency=freq_config.get('min_frequency', 0.6),
            group_column=freq_config.get('group_column'),
            per_group=freq_config.get('per_group', True)
        )
        filter_history.append(('frequency', freq_stats))
    
    # Apply score filter if configured
    if filter_config.get('score_filter', {}).get('enabled', False):
        score_config = filter_config['score_filter']
        current_df, score_stats = score_filter(
            current_df,
            score_method=score_config.get('method', 'total_abundance'),
            min_score_percentile=score_config.get('min_percentile', 50.0),
            weights=score_config.get('weights')
        )
        filter_history.append(('score', score_stats))
    
    # Apply custom filters if configured
    if filter_config.get('custom_filters', {}).get('enabled', False):
        custom_config = filter_config['custom_filters']
        current_df, custom_stats = apply_custom_filters(current_df, custom_config)
        filter_history.append(('custom', custom_stats))
    
    # Compile comprehensive statistics
    combined_stats = {
        'initial_features': len(feature_df.columns),
        'final_features': len(current_df.columns),
        'total_removed': len(feature_df.columns) - len(current_df.columns),
        'total_reduction_percentage': ((len(feature_df.columns) - len(current_df.columns)) / len(feature_df.columns)) * 100,
        'filter_sequence': filter_history,
        'configuration': filter_config
    }
    
    print(f"✅ Combined filtering complete!")
    print(f"  Initial: {len(feature_df.columns)} features")
    print(f"  Final: {len(current_df.columns)} features")
    print(f"  Total reduction: {combined_stats['total_reduction_percentage']:.1f}%")
    
    return current_df, combined_stats


def apply_custom_filters(
    feature_df: pd.DataFrame,
    custom_config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply custom filtering rules
    
    Args:
        feature_df: Feature matrix
        custom_config: Custom filter configuration
        
    Returns:
        Tuple of (filtered_df, custom_stats)
    """
    print("🛠️ Applying custom filters...")
    
    current_df = feature_df.copy()
    custom_operations = []
    
    # Example custom filters
    if custom_config.get('remove_low_variance', False):
        # Remove features with very low variance
        variance_threshold = custom_config.get('variance_threshold', 0.0)
        feature_variances = current_df.var(axis=0)
        high_variance_features = feature_variances[feature_variances > variance_threshold].index
        current_df = current_df[high_variance_features]
        
        custom_operations.append({
            'operation': 'remove_low_variance',
            'threshold': variance_threshold,
            'features_removed': len(feature_df.columns) - len(high_variance_features)
        })
    
    if custom_config.get('remove_outlier_features', False):
        # Remove features with extreme outlier values
        outlier_threshold = custom_config.get('outlier_zscore_threshold', 3.0)
        z_scores = np.abs((current_df - current_df.mean()) / current_df.std())
        non_outlier_features = z_scores.max(axis=0) <= outlier_threshold
        current_df = current_df.loc[:, non_outlier_features]
        
        custom_operations.append({
            'operation': 'remove_outlier_features',
            'threshold': outlier_threshold,
            'features_removed': (~non_outlier_features).sum()
        })
    
    stats = {
        'custom_operations': custom_operations,
        'features_before_custom': len(feature_df.columns),
        'features_after_custom': len(current_df.columns)
    }
    
    return current_df, stats


def load_filter_config(config_file: Union[str, Path]) -> Dict:
    """
    Load filter configuration from YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FilteringError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise FilteringError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def save_filter_report(
    stats: Dict,
    output_file: Path
) -> None:
    """
    Save filtering statistics to JSON file
    
    Args:
        stats: Filtering statistics
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"  📄 Filter report saved: {output_file}")


# Default configuration template
DEFAULT_FILTER_CONFIG = {
    'frequency_filter': {
        'enabled': True,
        'min_frequency': 0.6,
        'per_group': True,
        'group_column': None
    },
    'score_filter': {
        'enabled': True,
        'method': 'total_abundance',
        'min_percentile': 50.0,
        'weights': {
            'total_abundance': 0.4,
            'mean_intensity': 0.3,
            'presence_score': 0.2,
            'cv_score': 0.1
        }
    },
    'custom_filters': {
        'enabled': False,
        'remove_low_variance': True,
        'variance_threshold': 0.0,
        'remove_outlier_features': False,
        'outlier_zscore_threshold': 3.0
    }
}


# CLI interface
def main():
    """Command line interface for feature filtering"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Filtering")
    parser.add_argument("feature_matrix", help="Input feature matrix CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--config", help="Filter configuration file (YAML/JSON)")
    parser.add_argument("--frequency", type=float, default=0.6, help="Minimum frequency threshold")
    parser.add_argument("--score-percentile", type=float, default=50.0, help="Minimum score percentile")
    parser.add_argument("--score-method", default='total_abundance', 
                       choices=['total_abundance', 'mean_intensity', 'cv_score', 'presence_score', 'composite'],
                       help="Scoring method")
    
    args = parser.parse_args()
    
    try:
        # Load feature matrix
        print(f"Loading feature matrix from {args.feature_matrix}...")
        feature_df = pd.read_csv(args.feature_matrix, index_col=0)
        
        # Load or create configuration
        if args.config:
            filter_config = load_filter_config(args.config)
        else:
            # Use command line arguments
            filter_config = {
                'frequency_filter': {
                    'enabled': True,
                    'min_frequency': args.frequency,
                    'per_group': False
                },
                'score_filter': {
                    'enabled': True,
                    'method': args.score_method,
                    'min_percentile': args.score_percentile
                },
                'custom_filters': {'enabled': False}
            }
        
        # Run filtering
        filtered_df, stats = combined_filter(feature_df, filter_config)
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "filtered_features.csv"
        filtered_df.to_csv(output_file)
        
        report_file = output_dir / "filtering_report.json"
        save_filter_report(stats, report_file)
        
        # Save configuration for reference
        config_file = output_dir / "filter_config_used.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(filter_config, f, indent=2)
        
        print(f"\n✅ Filtering completed!")
        print(f"Output: {output_file}")
        print(f"Report: {report_file}")
        print(f"Config: {config_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 