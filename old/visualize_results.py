#!/usr/bin/env python3
"""
OmniBio Results Visualization
View and analyze your biomarker analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import requests

# Configuration
API_KEY = "omnibio-dev-key-12345"
BASE_URL = "http://localhost:8000"

def find_latest_analysis():
    """Find the most recent analysis results"""
    results_dir = Path("biomarker/api/results")
    if not results_dir.exists():
        print("❌ No results directory found")
        return None
    
    # Get all analysis directories
    analyses = [d for d in results_dir.iterdir() if d.is_dir()]
    if not analyses:
        print("❌ No analysis results found")
        return None
    
    # Sort by modification time (most recent first)
    latest = max(analyses, key=lambda x: x.stat().st_mtime)
    return latest

def load_statistical_results(analysis_dir):
    """Load statistical analysis results"""
    stats_file = analysis_dir / "statistical" / "statistical_results.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        print(f"✅ Loaded {len(df)} statistical results")
        return df
    else:
        print("❌ No statistical results file found")
        return None

def show_summary_stats(df):
    """Show summary statistics"""
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Basic stats
    print(f"Total features analyzed: {len(df)}")
    
    # Significance thresholds
    significant_raw = (df['p_value'] < 0.05).sum()
    significant_fdr = (df['adjusted_p_value'] < 0.05).sum()
    
    print(f"Significant (p < 0.05): {significant_raw}")
    print(f"Significant (FDR < 0.05): {significant_fdr}")
    
    # Effect sizes
    large_effect = (abs(df['log2_fold_change']) > 1).sum()
    print(f"Large effect size (|log2FC| > 1): {large_effect}")
    
    # Top features
    print(f"\n🔝 TOP SIGNIFICANT FEATURES:")
    top_features = df.nsmallest(5, 'adjusted_p_value')[['feature_name', 'log2_fold_change', 'p_value', 'adjusted_p_value']]
    for _, feature in top_features.iterrows():
        print(f"   {feature['feature_name']}: log2FC={feature['log2_fold_change']:.2f}, FDR={feature['adjusted_p_value']:.2e}")

def create_volcano_plot(df, save_path=None):
    """Create an enhanced volcano plot"""
    plt.figure(figsize=(12, 8))
    
    # Calculate -log10 p-values
    neg_log_p = -np.log10(df['p_value'])
    
    # Color points based on significance and effect size
    colors = []
    for _, row in df.iterrows():
        if row['adjusted_p_value'] < 0.05 and abs(row['log2_fold_change']) > 1:
            colors.append('red')  # Significant and large effect
        elif row['adjusted_p_value'] < 0.05:
            colors.append('orange')  # Significant but small effect
        elif abs(row['log2_fold_change']) > 1:
            colors.append('blue')  # Large effect but not significant
        else:
            colors.append('gray')  # Neither significant nor large effect
    
    # Create scatter plot
    plt.scatter(df['log2_fold_change'], neg_log_p, c=colors, alpha=0.6, s=30)
    
    # Add significance lines
    plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p = 0.05')
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=-1, color='black', linestyle='--', alpha=0.5)
    
    # Labels and title
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-Log10 P-value', fontsize=12)
    plt.title('Volcano Plot: Biomarker Analysis Results', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Significant & Large Effect'),
        Patch(facecolor='orange', label='Significant Only'),
        Patch(facecolor='blue', label='Large Effect Only'),
        Patch(facecolor='gray', label='Not Significant')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Enhanced volcano plot saved: {save_path}")
    
    plt.show()

def create_effect_size_distribution(df):
    """Plot distribution of effect sizes"""
    plt.figure(figsize=(10, 6))
    
    # Histogram of log2 fold changes
    plt.hist(df['log2_fold_change'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for effect size thresholds
    plt.axvline(x=1, color='red', linestyle='--', label='|log2FC| = 1')
    plt.axvline(x=-1, color='red', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='No change')
    
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Distribution of Effect Sizes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_significance_plot(df):
    """Plot p-value distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw p-values
    ax1.hist(df['p_value'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Raw P-value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adjusted p-values
    ax2.hist(df['adjusted_p_value'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(x=0.05, color='red', linestyle='--', label='FDR = 0.05')
    ax2.set_xlabel('Adjusted P-value (FDR)')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('FDR-Adjusted P-value Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_top_features(df, n_features=10):
    """Analyze top significant features"""
    print(f"\n🔬 TOP {n_features} BIOMARKERS")
    print("=" * 60)
    
    # Get top features by adjusted p-value
    top_features = df.nsmallest(n_features, 'adjusted_p_value')
    
    # Create a detailed table
    display_cols = ['feature_name', 'log2_fold_change', 'p_value', 'adjusted_p_value']
    if 'feature_description' in df.columns:
        display_cols.append('feature_description')
    
    for i, (_, feature) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {feature['feature_name']}")
        print(f"    Log2 Fold Change: {feature['log2_fold_change']:8.3f}")
        print(f"    P-value:          {feature['p_value']:8.2e}")
        print(f"    FDR:              {feature['adjusted_p_value']:8.2e}")
        
        # Interpretation
        direction = "↑ Up-regulated" if feature['log2_fold_change'] > 0 else "↓ Down-regulated"
        magnitude = "Large" if abs(feature['log2_fold_change']) > 1 else "Moderate"
        print(f"    Effect: {magnitude} {direction}")
        print()

def get_results_from_api(analysis_id):
    """Get results directly from API"""
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{BASE_URL}/analyses/{analysis_id}/results", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API request failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ API error: {e}")
        return None

def main():
    """Main visualization workflow"""
    print("🔬 OMNIBIO RESULTS VISUALIZATION")
    print("=" * 50)
    
    # Find latest analysis
    analysis_dir = find_latest_analysis()
    if not analysis_dir:
        return
    
    analysis_id = analysis_dir.name
    print(f"📁 Analyzing results: {analysis_id}")
    
    # Load statistical results
    df = load_statistical_results(analysis_dir)
    if df is None:
        return
    
    # Show summary statistics
    show_summary_stats(df)
    
    # Analyze top features
    analyze_top_features(df)
    
    # Create visualizations
    print("\n📊 GENERATING VISUALIZATIONS...")
    print("-" * 30)
    
    # Enhanced volcano plot
    volcano_path = f"enhanced_volcano_{analysis_id[:8]}.png"
    create_volcano_plot(df, volcano_path)
    
    # Effect size distribution
    create_effect_size_distribution(df)
    
    # Significance plots
    create_significance_plot(df)
    
    # Show file locations
    print(f"\n📁 RESULT FILES:")
    print(f"   Original results: {analysis_dir}")
    print(f"   Enhanced volcano: {volcano_path}")
    print(f"   Statistical data: {analysis_dir}/statistical/statistical_results.csv")
    print(f"   Original volcano: {analysis_dir}/statistical/volcano_plot.png")
    
    # API results
    print(f"\n🔗 API ACCESS:")
    print(f"   GET {BASE_URL}/analyses/{analysis_id}/results")
    print(f"   API Key: {API_KEY}")

if __name__ == "__main__":
    main() 