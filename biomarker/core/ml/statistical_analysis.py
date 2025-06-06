#!/usr/bin/env python3
"""
Statistical Analysis Module for OmniBio MVP
Implements Task #11: t-tests, volcano plots, and pathway analysis stubs
Based on run_univariate() functionality from biomarker_metabo.py
"""

import os
import sys
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
import warnings

# Optional plotly import (graceful fallback to matplotlib)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Using matplotlib for all plots.")


class StatisticalAnalysisError(Exception):
    """Raised when statistical analysis fails"""
    pass


def run_univariate_tests(
    df: pd.DataFrame, 
    labels: pd.Series,
    test_type: str = 'ttest',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run univariate statistical tests on features
    Based on run_univariate() from biomarker_metabo.py
    
    Args:
        df: Feature matrix (samples x features)
        labels: Sample labels (binary: Case/Control)
        test_type: Type of test ('ttest', 'mannwhitney')
        alpha: Significance threshold
        
    Returns:
        DataFrame with test results for each feature
    """
    print(f"Running univariate {test_type} tests...")
    
    # Align data and labels
    common_samples = df.index.intersection(labels.index)
    if len(common_samples) == 0:
        raise StatisticalAnalysisError("No common samples between features and labels")
    
    df_aligned = df.loc[common_samples]
    labels_aligned = labels.loc[common_samples]
    
    # Get unique groups
    unique_groups = labels_aligned.unique()
    if len(unique_groups) != 2:
        raise StatisticalAnalysisError(f"Expected 2 groups, got {len(unique_groups)}: {unique_groups}")
    
    group1, group2 = unique_groups
    group1_samples = df_aligned[labels_aligned == group1]
    group2_samples = df_aligned[labels_aligned == group2]
    
    print(f"  Group 1 ({group1}): {len(group1_samples)} samples")
    print(f"  Group 2 ({group2}): {len(group2_samples)} samples")
    
    results = []
    
    for feature in df_aligned.columns:
        try:
            # Get values for each group
            values1 = group1_samples[feature].dropna()
            values2 = group2_samples[feature].dropna()
            
            if len(values1) < 3 or len(values2) < 3:
                # Skip features with too few values
                continue
            
            # Calculate basic statistics
            mean1 = values1.mean()
            mean2 = values2.mean()
            std1 = values1.std()
            std2 = values2.std()
            
            # Calculate fold change (log2)
            # Add small constant to avoid division by zero
            fold_change = np.log2((mean1 + 1e-8) / (mean2 + 1e-8))
            
            # Handle NaN values in fold_change
            if np.isnan(fold_change) or np.isinf(fold_change):
                # Calculate simple difference as fallback
                fold_change = mean1 - mean2
            
            # Perform statistical test
            if test_type == 'ttest':
                statistic, p_value = ttest_ind(values1, values2, equal_var=False)
            elif test_type == 'mannwhitney':
                from scipy.stats import mannwhitneyu
                statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
            else:
                raise StatisticalAnalysisError(f"Unsupported test type: {test_type}")
            
            # Calculate -log10(p-value) for volcano plot
            neg_log_p = -np.log10(max(p_value, 1e-10))  # Avoid log(0)
            
            results.append({
                'feature': feature,
                'group1_mean': mean1,
                'group2_mean': mean2,
                'group1_std': std1,
                'group2_std': std2,
                'fold_change': fold_change,
                'statistic': statistic,
                'p_value': p_value,
                'neg_log10_p': neg_log_p,
                'significant': p_value < alpha,
                'n_group1': len(values1),
                'n_group2': len(values2)
            })
            
        except Exception as e:
            print(f"Warning: Failed to test feature {feature}: {e}")
            continue
    
    if len(results) == 0:
        raise StatisticalAnalysisError("No valid statistical tests performed")
    
    results_df = pd.DataFrame(results)
    
    # Replace any remaining NaN/inf values with None for JSON serialization
    results_df = results_df.replace([np.nan, np.inf, -np.inf], None)
    
    # Multiple testing correction (Benjamini-Hochberg)
    from scipy.stats import false_discovery_control
    if len(results_df) > 1:
        results_df['p_adjusted'] = false_discovery_control(results_df['p_value'])
        results_df['significant_adj'] = results_df['p_adjusted'] < alpha
    else:
        results_df['p_adjusted'] = results_df['p_value']
        results_df['significant_adj'] = results_df['significant']
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    print(f"  ✓ Completed {len(results_df)} tests")
    print(f"  ✓ Significant features (raw): {results_df['significant'].sum()}")
    print(f"  ✓ Significant features (adjusted): {results_df['significant_adj'].sum()}")
    
    return results_df


def generate_volcano_plot(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Volcano Plot",
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05
) -> Path:
    """
    Generate interactive volcano plot (if plotly available, otherwise skip)
    
    Args:
        results_df: Results from univariate tests
        output_path: Path to save HTML plot
        title: Plot title
        fc_threshold: Fold change threshold for coloring
        p_threshold: P-value threshold for coloring
        
    Returns:
        Path to saved plot (or None if plotly not available)
    """
    if not PLOTLY_AVAILABLE:
        print("  Skipping interactive volcano plot (plotly not available)")
        return None
    
    # Prepare data for plotting
    plot_df = results_df.copy()
    
    # Color points based on significance and fold change
    plot_df['color'] = 'Not Significant'
    plot_df.loc[(plot_df['p_value'] < p_threshold) & (plot_df['fold_change'] > fc_threshold), 'color'] = 'Up-regulated'
    plot_df.loc[(plot_df['p_value'] < p_threshold) & (plot_df['fold_change'] < -fc_threshold), 'color'] = 'Down-regulated'
    plot_df.loc[(plot_df['p_value'] < p_threshold) & (abs(plot_df['fold_change']) <= fc_threshold), 'color'] = 'Significant'
    
    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x='fold_change',
        y='neg_log10_p',
        color='color',
        hover_data=['feature', 'p_value', 'group1_mean', 'group2_mean'],
        color_discrete_map={
            'Up-regulated': 'red',
            'Down-regulated': 'blue', 
            'Significant': 'orange',
            'Not Significant': 'gray'
        },
        title=title,
        labels={
            'fold_change': 'Log2(Fold Change)',
            'neg_log10_p': '-Log10(P-value)'
        }
    )
    
    # Add threshold lines
    fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="black", opacity=0.5)
    
    # Customize layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    # Save as HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_path))
    
    print(f"✓ Interactive volcano plot saved: {output_path}")
    return output_path


def generate_static_volcano_plot(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "Volcano Plot",
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05
) -> Path:
    """
    Generate static volcano plot using matplotlib
    
    Args:
        results_df: Results from univariate tests
        output_path: Path to save PNG plot
        title: Plot title
        fc_threshold: Fold change threshold
        p_threshold: P-value threshold
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate data by significance
    not_sig = results_df[results_df['p_value'] >= p_threshold]
    sig_no_fc = results_df[(results_df['p_value'] < p_threshold) & (abs(results_df['fold_change']) <= fc_threshold)]
    up_reg = results_df[(results_df['p_value'] < p_threshold) & (results_df['fold_change'] > fc_threshold)]
    down_reg = results_df[(results_df['p_value'] < p_threshold) & (results_df['fold_change'] < -fc_threshold)]
    
    # Plot points
    ax.scatter(not_sig['fold_change'], not_sig['neg_log10_p'], 
               c='gray', alpha=0.6, s=20, label='Not Significant')
    ax.scatter(sig_no_fc['fold_change'], sig_no_fc['neg_log10_p'], 
               c='orange', alpha=0.7, s=20, label='Significant')
    ax.scatter(up_reg['fold_change'], up_reg['neg_log10_p'], 
               c='red', alpha=0.7, s=20, label='Up-regulated')
    ax.scatter(down_reg['fold_change'], down_reg['neg_log10_p'], 
               c='blue', alpha=0.7, s=20, label='Down-regulated')
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=fc_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Log2(Fold Change)')
    ax.set_ylabel('-Log10(P-value)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Static volcano plot saved: {output_path}")
    return output_path


def pathway_analysis_stub(
    results_df: pd.DataFrame,
    output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Placeholder for pathway analysis using mummichog
    As specified in the scoping document: "write stub that logs TODO"
    
    Args:
        results_df: Statistical test results
        output_dir: Directory for output files
        
    Returns:
        Dict with placeholder results
    """
    print("🔄 Pathway Analysis (Stub Implementation)")
    print("  TODO: Implement mummichog enrichment analysis")
    print("  TODO: Connect to pathway databases (KEGG, BioCyc)")
    print("  TODO: Generate pathway enrichment plots")
    print("  TODO: Export pathway results to JSON/CSV")
    
    # Create placeholder files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stub pathway results
    stub_results = {
        'status': 'stub_implementation',
        'todo_items': [
            'Implement mummichog pathway analysis',
            'Add pathway database connections',
            'Generate enrichment visualizations',
            'Export pathway results'
        ],
        'input_features': len(results_df),
        'significant_features': results_df['significant'].sum(),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save stub results
    import json
    stub_file = output_dir / "pathway_analysis_stub.json"
    with open(stub_file, 'w') as f:
        json.dump(stub_results, f, indent=2)
    
    print(f"  ✓ Stub results saved: {stub_file}")
    return stub_results


def run_complete_statistical_analysis(
    df: pd.DataFrame,
    labels: pd.Series,
    output_dir: Union[str, Path],
    test_type: str = 'ttest',
    alpha: float = 0.05,
    fc_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Run complete statistical analysis pipeline
    
    Args:
        df: Feature matrix
        labels: Sample labels
        output_dir: Output directory
        test_type: Statistical test type
        alpha: Significance threshold
        fc_threshold: Fold change threshold
        
    Returns:
        Dict with all results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("STATISTICAL ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Step 1: Univariate tests
    results_df = run_univariate_tests(df, labels, test_type, alpha)
    
    # Save results
    results_file = output_dir / "statistical_results.csv"
    results_df.to_csv(results_file, index=False)
    
    # Step 2: Generate volcano plots
    volcano_html = generate_volcano_plot(
        results_df, 
        output_dir / "volcano_plot.html",
        fc_threshold=fc_threshold,
        p_threshold=alpha
    )
    
    volcano_png = generate_static_volcano_plot(
        results_df,
        output_dir / "volcano_plot.png", 
        fc_threshold=fc_threshold,
        p_threshold=alpha
    )
    
    # Step 3: Pathway analysis stub
    pathway_results = pathway_analysis_stub(results_df, output_dir)
    
    # Summary
    summary = {
        'analysis_type': test_type,
        'n_features_tested': len(results_df),
        'n_significant_raw': int(results_df['significant'].sum()),
        'n_significant_adj': int(results_df['significant_adj'].sum()),
        'alpha': alpha,
        'fc_threshold': fc_threshold,
        'top_features': results_df.head(10)[['feature', 'p_value', 'fold_change']].to_dict('records')
    }
    
    # Prepare files dict (excluding None values)
    files_dict = {
        'results_csv': results_file,
        'volcano_png': volcano_png,
        'pathway_stub': output_dir / "pathway_analysis_stub.json"
    }
    
    # Add interactive plot if available
    if volcano_html is not None:
        files_dict['volcano_html'] = volcano_html
    
    return {
        'results_df': results_df,
        'summary': summary,
        'files': files_dict,
        'pathway_results': pathway_results
    }


# CLI interface
def main():
    """Command line interface for statistical analysis"""
    if len(sys.argv) < 2:
        print("Usage: python statistical_analysis.py <mwtab_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "statistical_output"
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        
        # Load data
        print(f"Loading data from {input_file}...")
        df, metadata = load_file(input_file)
        
        # Extract labels
        print("Extracting labels...")
        try:
            import mwtab
            mw = next(mwtab.read_files(input_file))
            ssf = pd.DataFrame(mw['SUBJECT_SAMPLE_FACTORS'])
            factors = pd.json_normalize(ssf['Factors'])
            ssf = ssf.drop(columns='Factors').join(factors)
            labels = ssf.set_index('Sample ID')['Group'].reindex(df.index)
        except Exception as e:
            print(f"Warning: Using dummy labels: {e}")
            n_samples = len(df)
            labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                             index=df.index, name='Group')
        
        # Run analysis
        analysis_results = run_complete_statistical_analysis(df, labels, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        summary = analysis_results['summary']
        print(f"Features tested: {summary['n_features_tested']}")
        print(f"Significant (raw p < {summary['alpha']}): {summary['n_significant_raw']}")
        print(f"Significant (adjusted): {summary['n_significant_adj']}")
        
        print(f"\nTop 5 significant features:")
        for i, feat in enumerate(summary['top_features'][:5]):
            print(f"  {i+1}. {feat['feature']}: p={feat['p_value']:.2e}, FC={feat['fold_change']:.2f}")
        
        print(f"\nOutput files:")
        for name, path in analysis_results['files'].items():
            print(f"  {name}: {path}")
        
        print("\n✅ Statistical analysis completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 