#!/usr/bin/env python3
"""
Statistical Analysis Module for OmniBio MVP (Simplified Version)
Implements Task #11: t-tests, volcano plots, and pathway analysis stubs
Based on run_univariate() functionality from biomarker_metabo.py
Uses only matplotlib (no plotly dependencies)
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
    Generate volcano plot using matplotlib
    
    Args:
        results_df: Results from univariate tests
        output_path: Path to save PNG plot
        title: Plot title
        fc_threshold: Fold change threshold
        p_threshold: P-value threshold
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Separate data by significance
    not_sig = results_df[results_df['p_value'] >= p_threshold]
    sig_no_fc = results_df[(results_df['p_value'] < p_threshold) & (abs(results_df['fold_change']) <= fc_threshold)]
    up_reg = results_df[(results_df['p_value'] < p_threshold) & (results_df['fold_change'] > fc_threshold)]
    down_reg = results_df[(results_df['p_value'] < p_threshold) & (results_df['fold_change'] < -fc_threshold)]
    
    # Plot points with different colors and proper legend labels
    if len(not_sig) > 0:
        ax.scatter(not_sig['fold_change'], not_sig['neg_log10_p'], 
                   c='#94A3B8', alpha=0.6, s=20, label=f'Not Significant (n={len(not_sig)})', 
                   edgecolors='white', linewidth=0.5)
    if len(sig_no_fc) > 0:
        ax.scatter(sig_no_fc['fold_change'], sig_no_fc['neg_log10_p'], 
                   c='#F59E0B', alpha=0.8, s=25, label=f'Significant Only (n={len(sig_no_fc)})',
                   edgecolors='white', linewidth=0.5)
    if len(up_reg) > 0:
        ax.scatter(up_reg['fold_change'], up_reg['neg_log10_p'], 
                   c='#EF4444', alpha=0.8, s=25, label=f'Up-regulated (n={len(up_reg)})',
                   edgecolors='white', linewidth=0.5)
    if len(down_reg) > 0:
        ax.scatter(down_reg['fold_change'], down_reg['neg_log10_p'], 
                   c='#3B82F6', alpha=0.8, s=25, label=f'Down-regulated (n={len(down_reg)})',
                   edgecolors='white', linewidth=0.5)
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='black', linestyle='--', alpha=0.5, 
               linewidth=1, label=f'p-value threshold ({p_threshold})')
    ax.axvline(x=fc_threshold, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add labels for threshold lines
    ax.text(fc_threshold + 0.1, ax.get_ylim()[1] * 0.9, f'FC = {fc_threshold}', 
            rotation=90, va='top', fontsize=9, alpha=0.7)
    ax.text(-fc_threshold - 0.1, ax.get_ylim()[1] * 0.9, f'FC = -{fc_threshold}', 
            rotation=90, va='top', fontsize=9, alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel('Log₂ Fold Change (Case vs Control)', fontsize=14, fontweight='bold')
    ax.set_ylabel('-Log₁₀(p-value)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add annotation box with summary statistics
    total_features = len(results_df)
    total_sig = results_df['significant'].sum()
    total_sig_adj = results_df['significant_adj'].sum()
    
    summary_text = (f'Total Features: {total_features}\n'
                   f'Significant (raw p<{p_threshold}): {total_sig}\n'
                   f'Significant (FDR<{p_threshold}): {total_sig_adj}\n'
                   f'Fold Change Threshold: ±{fc_threshold}')
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightblue', alpha=0.8), fontsize=10)
    
    # Add color explanation in the legend area
    legend_text = ('Gray: Not significant\n'
                  'Orange: Significant, small effect\n'
                  'Red: Upregulated & significant\n'
                  'Blue: Downregulated & significant')
    
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8), 
            fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Volcano plot saved: {output_path}")
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
        'significant_features': int(results_df['significant'].sum()),
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
    
    # Step 2: Generate volcano plot
    volcano_png = generate_volcano_plot(
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
    
    return {
        'results_df': results_df,
        'summary': summary,
        'files': {
            'results_csv': results_file,
            'volcano_png': volcano_png,
            'pathway_stub': output_dir / "pathway_analysis_stub.json"
        },
        'pathway_results': pathway_results
    }


# CLI interface
def main():
    """Command line interface for statistical analysis"""
    if len(sys.argv) < 2:
        print("Usage: python statistical_analysis_simple.py <mwtab_file> [output_dir]")
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