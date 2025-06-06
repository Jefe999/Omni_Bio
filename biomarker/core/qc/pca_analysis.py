#!/usr/bin/env python3
"""
PCA Analysis Module for OmniBio MVP
Implements Task #4: Batch-effect explorer with PCA plots
Build feature matrix using simple binning for QC, plot PCA colored by Class
"""

import os
import sys
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


class PCAAnalysisError(Exception):
    """Raised when PCA analysis fails"""
    pass


def preprocess_for_pca(df: pd.DataFrame, fill_missing: str = 'median') -> pd.DataFrame:
    """
    Preprocess data for PCA analysis
    
    Args:
        df: Feature matrix (samples x features)
        fill_missing: Method for handling missing values ('median', 'mean', 'zero')
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"Preprocessing data for PCA...")
    
    # Make a copy
    df_clean = df.copy()
    
    # Remove features with too many missing values (>50%)
    missing_fraction = df_clean.isna().sum() / len(df_clean)
    features_to_keep = missing_fraction[missing_fraction <= 0.5].index
    df_clean = df_clean[features_to_keep]
    
    n_removed = len(df.columns) - len(features_to_keep)
    if n_removed > 0:
        print(f"  - Removed {n_removed} features with >50% missing values")
    
    # Handle missing values
    if fill_missing == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif fill_missing == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif fill_missing == 'zero':
        df_clean = df_clean.fillna(0)
    else:
        raise PCAAnalysisError(f"Unsupported fill_missing method: {fill_missing}")
    
    # Remove features with zero variance
    initial_features = df_clean.shape[1]
    df_clean = df_clean.loc[:, df_clean.var() > 0]
    zero_var_removed = initial_features - df_clean.shape[1]
    
    if zero_var_removed > 0:
        print(f"  - Removed {zero_var_removed} features with zero variance")
    
    print(f"  ✓ Preprocessed data: {df_clean.shape[0]} samples, {df_clean.shape[1]} features")
    return df_clean


def run_pca_analysis(
    df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    n_components: int = 10,
    scale_data: bool = True
) -> Dict[str, Any]:
    """
    Run PCA analysis on feature matrix
    
    Args:
        df: Feature matrix (samples x features)
        labels: Optional sample labels for coloring
        n_components: Number of PCA components to compute
        scale_data: Whether to scale data before PCA
        
    Returns:
        Dict with PCA results
    """
    print(f"Running PCA analysis...")
    
    # Preprocess data
    df_processed = preprocess_for_pca(df)
    
    if len(df_processed.columns) == 0:
        raise PCAAnalysisError("No features remaining after preprocessing")
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_processed)
        feature_names = list(df_processed.columns)
    else:
        X_scaled = df_processed.values
        scaler = None
        feature_names = list(df_processed.columns)
    
    # Fit PCA
    n_components = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create results DataFrame
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pc_columns, index=df_processed.index)
    
    # Add labels if provided
    if labels is not None:
        aligned_labels = labels.reindex(pca_df.index)
        pca_df['Group'] = aligned_labels
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"  ✓ PCA completed: {n_components} components")
    print(f"  ✓ PC1 explains {explained_var[0]:.1%} variance")
    print(f"  ✓ PC1+PC2 explain {cumulative_var[1]:.1%} variance")
    
    return {
        'pca_df': pca_df,
        'pca_model': pca,
        'scaler': scaler,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'feature_names': feature_names,
        'n_samples': X_scaled.shape[0],
        'n_features': X_scaled.shape[1]
    }


def generate_pca_scree_plot(
    pca_results: Dict[str, Any],
    output_path: Union[str, Path],
    max_components: int = 10
) -> Path:
    """
    Generate scree plot showing explained variance
    
    Args:
        pca_results: Results from run_pca_analysis
        output_path: Path to save plot
        max_components: Maximum components to show
        
    Returns:
        Path to saved plot
    """
    explained_var = pca_results['explained_variance'][:max_components]
    cumulative_var = pca_results['cumulative_variance'][:max_components]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var * 100)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('Scree Plot - Individual Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 'o-')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80%')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Scree plot saved: {output_path}")
    return output_path


def generate_interactive_pca_plot(
    pca_results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "PCA Analysis - Batch Effect Explorer"
) -> Path:
    """
    Generate interactive PCA plot colored by class
    
    Args:
        pca_results: Results from run_pca_analysis
        output_path: Path to save HTML plot
        title: Plot title
        
    Returns:
        Path to saved plot
    """
    pca_df = pca_results['pca_df']
    explained_var = pca_results['explained_variance']
    
    # Determine color column
    color_col = 'Group' if 'Group' in pca_df.columns else None
    
    # Create interactive plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=color_col,
        hover_data=list(pca_df.columns),
        title=title,
        labels={
            'PC1': f'PC1 ({explained_var[0]:.1%} variance)',
            'PC2': f'PC2 ({explained_var[1]:.1%} variance)'
        }
    )
    
    # Customize layout
    fig.update_layout(
        width=800,
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    # Add loading plot if we have enough components
    if len(explained_var) >= 3:
        # Add 3D option as subplot
        pass  # Could add 3D plot here
    
    # Save as HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_path))
    
    print(f"✓ Interactive PCA plot saved: {output_path}")
    return output_path


def generate_static_pca_plot(
    pca_results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "PCA Analysis - Batch Effect Explorer"
) -> Path:
    """
    Generate static PCA plot using matplotlib
    
    Args:
        pca_results: Results from run_pca_analysis
        output_path: Path to save PNG plot
        title: Plot title
        
    Returns:
        Path to saved plot
    """
    pca_df = pca_results['pca_df']
    explained_var = pca_results['explained_variance']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by group if available
    if 'Group' in pca_df.columns:
        groups = pca_df['Group'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
        
        for i, group in enumerate(groups):
            mask = pca_df['Group'] == group
            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                      c=[colors[i]], label=str(group), alpha=0.7, s=50)
        ax.legend()
    else:
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add zero lines
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Static PCA plot saved: {output_path}")
    return output_path


def generate_loadings_plot(
    pca_results: Dict[str, Any],
    output_path: Union[str, Path],
    n_features: int = 20
) -> Path:
    """
    Generate loadings plot showing feature contributions
    
    Args:
        pca_results: Results from run_pca_analysis
        output_path: Path to save plot
        n_features: Number of top features to show
        
    Returns:
        Path to saved plot
    """
    pca_model = pca_results['pca_model']
    feature_names = pca_results['feature_names']
    
    # Get loadings for PC1 and PC2
    loadings = pca_model.components_[:2].T  # Features x PCs
    
    # Calculate magnitude of loadings
    loading_magnitude = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    
    # Get top features by loading magnitude
    top_indices = np.argsort(loading_magnitude)[-n_features:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot loadings as arrows
    for i in top_indices:
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.7)
        
        # Add feature label
        ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, 
               feature_names[i], fontsize=8, ha='center', va='center')
    
    # Add circle for reference
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_xlabel('PC1 Loadings')
    ax.set_ylabel('PC2 Loadings')
    ax.set_title(f'PCA Loadings Plot - Top {n_features} Features')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set reasonable limits
    max_val = np.max(np.abs(loadings[top_indices])) * 1.3
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Loadings plot saved: {output_path}")
    return output_path


def run_complete_pca_analysis(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    labels: Optional[pd.Series] = None,
    n_components: int = 10
) -> Dict[str, Any]:
    """
    Run complete PCA analysis pipeline
    
    Args:
        df: Feature matrix
        output_dir: Output directory
        labels: Sample labels (optional)
        n_components: Number of components
        
    Returns:
        Dict with all results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("PCA BATCH EFFECT ANALYSIS")
    print("=" * 50)
    
    # Run PCA analysis
    pca_results = run_pca_analysis(df, labels, n_components)
    
    # Save PCA scores
    scores_file = output_dir / "pca_scores.csv"
    pca_results['pca_df'].to_csv(scores_file)
    
    # Generate plots
    scree_plot = generate_pca_scree_plot(pca_results, output_dir / "pca_scree.png")
    
    interactive_plot = generate_interactive_pca_plot(
        pca_results, 
        output_dir / "pca_interactive.html"
    )
    
    static_plot = generate_static_pca_plot(
        pca_results,
        output_dir / "pca_plot.png"
    )
    
    loadings_plot = generate_loadings_plot(
        pca_results,
        output_dir / "pca_loadings.png"
    )
    
    # Summary
    summary = {
        'n_samples': pca_results['n_samples'],
        'n_features': pca_results['n_features'],
        'n_components': len(pca_results['explained_variance']),
        'pc1_variance': float(pca_results['explained_variance'][0]),
        'pc2_variance': float(pca_results['explained_variance'][1]),
        'total_variance_2pc': float(pca_results['cumulative_variance'][1]),
        'has_labels': 'Group' in pca_results['pca_df'].columns
    }
    
    return {
        'pca_results': pca_results,
        'summary': summary,
        'files': {
            'scores_csv': scores_file,
            'scree_plot': scree_plot,
            'interactive_plot': interactive_plot,
            'static_plot': static_plot,
            'loadings_plot': loadings_plot
        }
    }


# CLI interface
def main():
    """Command line interface for PCA analysis"""
    if len(sys.argv) < 2:
        print("Usage: python pca_analysis.py <mwtab_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pca_output"
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        
        # Load data
        print(f"Loading data from {input_file}...")
        df, metadata = load_file(input_file)
        
        # Extract labels if available
        labels = None
        print("Extracting labels...")
        try:
            import mwtab
            mw = next(mwtab.read_files(input_file))
            ssf = pd.DataFrame(mw['SUBJECT_SAMPLE_FACTORS'])
            factors = pd.json_normalize(ssf['Factors'])
            ssf = ssf.drop(columns='Factors').join(factors)
            labels = ssf.set_index('Sample ID')['Group'].reindex(df.index)
            print(f"✓ Found labels: {labels.value_counts().to_dict()}")
        except Exception as e:
            print(f"Warning: No labels found, proceeding without: {e}")
        
        # Run PCA analysis
        analysis_results = run_complete_pca_analysis(df, output_dir, labels)
        
        # Print summary
        print("\n" + "="*50)
        print("PCA ANALYSIS SUMMARY")
        print("="*50)
        
        summary = analysis_results['summary']
        print(f"Samples: {summary['n_samples']}")
        print(f"Features: {summary['n_features']}")
        print(f"PC1 variance: {summary['pc1_variance']:.1%}")
        print(f"PC2 variance: {summary['pc2_variance']:.1%}")
        print(f"Total (PC1+PC2): {summary['total_variance_2pc']:.1%}")
        print(f"Has group labels: {summary['has_labels']}")
        
        print(f"\nOutput files:")
        for name, path in analysis_results['files'].items():
            print(f"  {name}: {path}")
        
        print("\n✅ PCA analysis completed!")
        print(f"📊 View interactive plot: file://{Path(output_dir).absolute()}/pca_interactive.html")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 