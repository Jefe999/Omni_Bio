#!/usr/bin/env python3
"""
Task #6: Duplicate Feature Removal
MS-FLO logic for clustering and removing duplicate features.

Target: Reduce duplicate count by ≥90% while keeping highest abundance features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path
import json


class DeduplicationError(Exception):
    """Raised when deduplication fails"""
    pass


def deduplicate_features(
    feature_df: pd.DataFrame,
    mz_tolerance_ppm: float = 5.0,
    rt_tolerance_min: float = 0.1,
    min_samples_present: int = 1,
    keep_strategy: str = 'highest_abundance'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove duplicate features using MS-FLO clustering logic
    
    Args:
        feature_df: Feature matrix (samples × features) with feature metadata
        mz_tolerance_ppm: Mass tolerance in ppm for clustering
        rt_tolerance_min: Retention time tolerance in minutes
        min_samples_present: Minimum samples a feature must be present in
        keep_strategy: Strategy for keeping features ('highest_abundance', 'most_frequent')
        
    Returns:
        Tuple of (deduplicated_df, deduplication_stats)
        
    Raises:
        DeduplicationError: If feature metadata is missing or invalid
    """
    print(f"🔄 Starting feature deduplication...")
    print(f"  Input: {feature_df.shape[0]} samples × {feature_df.shape[1]} features")
    
    # Extract feature metadata
    if 'feature_metadata' not in feature_df.attrs:
        raise DeduplicationError("Feature metadata not found. Run OpenMS feature extraction first.")
    
    feature_metadata = feature_df.attrs['feature_metadata']
    
    # Validate required metadata columns
    required_cols = ['mz', 'rt']
    missing_cols = [col for col in required_cols if col not in feature_metadata.columns]
    if missing_cols:
        raise DeduplicationError(f"Missing metadata columns: {missing_cols}")
    
    # Filter features by sample presence
    initial_features = len(feature_df.columns)
    sample_presence = (feature_df > 0).sum(axis=0)
    valid_features = sample_presence >= min_samples_present
    
    filtered_df = feature_df.loc[:, valid_features]
    filtered_metadata = feature_metadata.loc[valid_features]
    
    print(f"  After sample presence filter (≥{min_samples_present}): {filtered_df.shape[1]} features")
    
    if len(filtered_df.columns) == 0:
        print("  ⚠️ No features remaining after filtering")
        return filtered_df, {'duplicates_removed': 0, 'clusters_found': 0}
    
    # Cluster features by m/z and RT similarity
    clusters = cluster_features_by_similarity(
        filtered_metadata,
        mz_tolerance_ppm,
        rt_tolerance_min
    )
    
    # Select representative features from each cluster
    deduplicated_df, stats = select_cluster_representatives(
        filtered_df,
        filtered_metadata,
        clusters,
        keep_strategy
    )
    
    # Update statistics
    stats.update({
        'initial_features': initial_features,
        'after_presence_filter': len(filtered_df.columns),
        'final_features': len(deduplicated_df.columns),
        'reduction_percentage': (1 - len(deduplicated_df.columns) / initial_features) * 100,
        'parameters': {
            'mz_tolerance_ppm': mz_tolerance_ppm,
            'rt_tolerance_min': rt_tolerance_min,
            'min_samples_present': min_samples_present,
            'keep_strategy': keep_strategy
        }
    })
    
    print(f"  ✅ Deduplication complete!")
    print(f"    Final: {deduplicated_df.shape[1]} features")
    print(f"    Removed: {stats['duplicates_removed']} duplicates ({stats['reduction_percentage']:.1f}% reduction)")
    print(f"    Clusters: {stats['clusters_found']}")
    
    return deduplicated_df, stats


def cluster_features_by_similarity(
    feature_metadata: pd.DataFrame,
    mz_tolerance_ppm: float,
    rt_tolerance_min: float
) -> np.ndarray:
    """
    Cluster features based on m/z and RT similarity using hierarchical clustering
    
    Args:
        feature_metadata: DataFrame with 'mz' and 'rt' columns
        mz_tolerance_ppm: Mass tolerance in ppm
        rt_tolerance_min: RT tolerance in minutes
        
    Returns:
        Array of cluster labels for each feature
    """
    n_features = len(feature_metadata)
    
    if n_features <= 1:
        return np.array([0] * n_features)
    
    # Extract m/z and RT values
    mz_values = feature_metadata['mz'].values
    rt_values = feature_metadata['rt'].values / 60.0  # Convert to minutes if in seconds
    
    # Create normalized coordinate matrix
    # Normalize m/z by relative tolerance and RT by absolute tolerance
    normalized_coords = np.column_stack([
        mz_values / (mz_tolerance_ppm * 1e-6 * mz_values),  # ppm-normalized m/z
        rt_values / rt_tolerance_min  # minute-normalized RT
    ])
    
    # Calculate pairwise distances
    distances = pdist(normalized_coords, metric='euclidean')
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='single')  # Single linkage for tight clusters
    
    # Form clusters with distance threshold of 1.0
    # (features within tolerance will have distance < 1.0)
    cluster_labels = fcluster(linkage_matrix, t=1.0, criterion='distance')
    
    return cluster_labels


def select_cluster_representatives(
    feature_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    keep_strategy: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select representative features from each cluster
    
    Args:
        feature_df: Feature intensity matrix
        feature_metadata: Feature metadata
        cluster_labels: Cluster assignment for each feature
        keep_strategy: Strategy for selecting representatives
        
    Returns:
        Tuple of (deduplicated_df, stats_dict)
    """
    unique_clusters = np.unique(cluster_labels)
    selected_features = []
    cluster_stats = []
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_features = feature_df.columns[cluster_mask]
        
        if len(cluster_features) == 1:
            # Single feature in cluster - keep it
            selected_features.extend(cluster_features)
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'n_features': 1,
                'selected_feature': cluster_features[0],
                'selection_reason': 'single_feature'
            })
        else:
            # Multiple features - select representative
            representative = select_representative_feature(
                feature_df[cluster_features],
                feature_metadata.loc[cluster_features],
                keep_strategy
            )
            selected_features.append(representative)
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'n_features': len(cluster_features),
                'selected_feature': representative,
                'selection_reason': keep_strategy,
                'cluster_features': list(cluster_features)
            })
    
    # Create deduplicated DataFrame
    deduplicated_df = feature_df[selected_features].copy()
    
    # Update metadata
    selected_metadata = feature_metadata.loc[selected_features].copy()
    deduplicated_df.attrs['feature_metadata'] = selected_metadata
    
    # Calculate statistics
    duplicates_removed = len(feature_df.columns) - len(selected_features)
    clusters_with_duplicates = sum(1 for stat in cluster_stats if stat['n_features'] > 1)
    
    stats = {
        'duplicates_removed': duplicates_removed,
        'clusters_found': len(unique_clusters),
        'clusters_with_duplicates': clusters_with_duplicates,
        'cluster_details': cluster_stats
    }
    
    return deduplicated_df, stats


def select_representative_feature(
    cluster_df: pd.DataFrame,
    cluster_metadata: pd.DataFrame,
    strategy: str
) -> str:
    """
    Select the representative feature from a cluster
    
    Args:
        cluster_df: Feature intensities for cluster members
        cluster_metadata: Metadata for cluster members
        strategy: Selection strategy
        
    Returns:
        Feature ID of the selected representative
    """
    if strategy == 'highest_abundance':
        # Select feature with highest total abundance
        total_abundance = cluster_df.sum(axis=0)
        return total_abundance.idxmax()
    
    elif strategy == 'most_frequent':
        # Select feature present in most samples
        presence_count = (cluster_df > 0).sum(axis=0)
        return presence_count.idxmax()
    
    elif strategy == 'highest_median':
        # Select feature with highest median intensity
        median_intensity = cluster_df.median(axis=0)
        return median_intensity.idxmax()
    
    else:
        # Default to highest abundance
        total_abundance = cluster_df.sum(axis=0)
        return total_abundance.idxmax()


def analyze_duplicates(
    feature_metadata: pd.DataFrame,
    mz_tolerance_ppm: float = 5.0,
    rt_tolerance_min: float = 0.1
) -> Dict:
    """
    Analyze potential duplicate features without removing them
    
    Args:
        feature_metadata: Feature metadata with m/z and RT
        mz_tolerance_ppm: Mass tolerance for duplicate detection
        rt_tolerance_min: RT tolerance for duplicate detection
        
    Returns:
        Dictionary with duplicate analysis results
    """
    cluster_labels = cluster_features_by_similarity(
        feature_metadata,
        mz_tolerance_ppm,
        rt_tolerance_min
    )
    
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    
    duplicate_clusters = cluster_counts[cluster_counts > 1]
    total_duplicates = np.sum(duplicate_clusters - 1)  # Extra features beyond one per cluster
    
    analysis = {
        'total_features': len(feature_metadata),
        'unique_clusters': len(unique_clusters),
        'clusters_with_duplicates': len(duplicate_clusters),
        'estimated_duplicates': int(total_duplicates),
        'duplicate_percentage': (total_duplicates / len(feature_metadata)) * 100,
        'largest_cluster_size': int(np.max(cluster_counts)),
        'parameters': {
            'mz_tolerance_ppm': mz_tolerance_ppm,
            'rt_tolerance_min': rt_tolerance_min
        }
    }
    
    return analysis


def save_deduplication_report(
    stats: Dict,
    output_file: Path
) -> None:
    """
    Save deduplication statistics to JSON file
    
    Args:
        stats: Deduplication statistics
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"  📄 Deduplication report saved: {output_file}")


# CLI interface
def main():
    """Command line interface for feature deduplication"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Deduplication")
    parser.add_argument("feature_matrix", help="Input feature matrix CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--mz-tolerance-ppm", type=float, default=5.0, help="m/z tolerance (ppm)")
    parser.add_argument("--rt-tolerance-min", type=float, default=0.1, help="RT tolerance (min)")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum samples present")
    parser.add_argument("--strategy", choices=['highest_abundance', 'most_frequent', 'highest_median'], 
                       default='highest_abundance', help="Representative selection strategy")
    
    args = parser.parse_args()
    
    try:
        # Load feature matrix
        print(f"Loading feature matrix from {args.feature_matrix}...")
        feature_df = pd.read_csv(args.feature_matrix, index_col=0)
        
        # For testing, create dummy metadata if not present
        if 'feature_metadata' not in feature_df.attrs:
            print("  Creating dummy metadata for testing...")
            n_features = len(feature_df.columns)
            dummy_metadata = pd.DataFrame({
                'mz': np.random.uniform(100, 1000, n_features),
                'rt': np.random.uniform(0, 1800, n_features)  # seconds
            }, index=feature_df.columns)
            feature_df.attrs['feature_metadata'] = dummy_metadata
        
        # Run deduplication
        deduplicated_df, stats = deduplicate_features(
            feature_df,
            mz_tolerance_ppm=args.mz_tolerance_ppm,
            rt_tolerance_min=args.rt_tolerance_min,
            min_samples_present=args.min_samples,
            keep_strategy=args.strategy
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "deduplicated_features.csv"
        deduplicated_df.to_csv(output_file)
        
        report_file = output_dir / "deduplication_report.json"
        save_deduplication_report(stats, report_file)
        
        print(f"\n✅ Deduplication completed!")
        print(f"Output: {output_file}")
        print(f"Report: {report_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 