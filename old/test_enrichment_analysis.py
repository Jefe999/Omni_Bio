#!/usr/bin/env python3
"""
Test script for Task #13: Pathway Enrichment Analysis

Tests the complete pathway enrichment pipeline with demo data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add biomarker package to path
sys.path.append('.')

from biomarker.core.analysis.pathway_analysis import (
    run_pathway_enrichment,
    PathwayEnrichmentParams
)


def create_demo_feature_matrix():
    """Create a demo feature matrix with metabolomics-like data"""
    np.random.seed(42)
    
    # Create feature names that could map to known compounds
    feature_names = [
        'mz_180.0634_rt_120.5',  # Glucose-like
        'mz_146.0579_rt_85.2',   # Pyruvate-like  
        'mz_117.0193_rt_95.8',   # Glutamate-like
        'mz_132.0532_rt_110.3',  # Ketoglutarate-like
        'mz_174.0528_rt_88.7',   # Oxaloacetate-like
        'mz_147.0532_rt_92.1',   # Aspartate-like
        'mz_204.0892_rt_105.6',  # Citrate-like
        'mz_166.0528_rt_78.9',   # Fumarate-like
        'mz_150.0528_rt_82.4',   # Malate-like
        'mz_181.0345_rt_73.2',   # Succinate-like
    ] + [f'feature_{i:06d}' for i in range(10, 100)]  # Additional random features
    
    # Create sample data (NAFLD study-like)
    sample_names = ['Control_' + str(i) for i in range(1, 26)] + ['NAFLD_' + str(i) for i in range(1, 26)]
    
    # Generate feature matrix
    data = np.random.lognormal(mean=3, sigma=1, size=(len(sample_names), len(feature_names)))
    
    # Add some missing values
    missing_mask = np.random.random(data.shape) < 0.05
    data[missing_mask] = 0
    
    feature_df = pd.DataFrame(data, index=sample_names, columns=feature_names)
    
    # Create metadata for features that look like real metabolites
    feature_metadata = pd.DataFrame({
        'mz': [float(name.split('_')[1]) if 'mz_' in name else np.random.uniform(100, 1000) for name in feature_names],
        'rt': [float(name.split('_')[3]) if 'rt_' in name else np.random.uniform(0, 1800) for name in feature_names]
    }, index=feature_names)
    
    # Attach metadata to DataFrame
    feature_df.attrs['feature_metadata'] = feature_metadata
    
    return feature_df


def main():
    """Test pathway enrichment analysis"""
    print("🧪 Testing Pathway Enrichment Analysis (Task #13)")
    print("=" * 60)
    
    # Create demo data
    print("📊 Creating demo feature matrix...")
    feature_df = create_demo_feature_matrix()
    print(f"  Created matrix: {feature_df.shape[0]} samples × {feature_df.shape[1]} features")
    
    # Define significant features (simulate differential analysis results)
    # Include the metabolite-like features as "significant"
    significant_features = [
        'mz_180.0634_rt_120.5',  # Glucose-like
        'mz_146.0579_rt_85.2',   # Pyruvate-like
        'mz_117.0193_rt_95.8',   # Glutamate-like
        'mz_132.0532_rt_110.3',  # Ketoglutarate-like
        'mz_204.0892_rt_105.6',  # Citrate-like
        'feature_000015',        # Random feature
        'feature_000032',        # Random feature
    ]
    
    print(f"  Significant features: {len(significant_features)}")
    
    # Test with default parameters
    print("\n🔬 Testing with default parameters...")
    params = PathwayEnrichmentParams()
    
    results = run_pathway_enrichment(
        feature_df,
        significant_features,
        params=params
    )
    
    # Print results summary
    print(f"\n📊 ENRICHMENT ANALYSIS RESULTS")
    print("=" * 40)
    
    print(f"Feature mapping rate: {results['feature_mapping']['mapping_rate']:.1%}")
    print(f"Mapped significant features: {len(results['feature_mapping']['significant_mapped'])}")
    
    summary = results['summary']
    print(f"\nTotal significant pathways: {summary['total_significant_pathways']}")
    
    for db_name, db_summary in summary['enrichment_overview'].items():
        print(f"{db_name.upper()}: {db_summary['significant']} / {db_summary['tested']} pathways significant")
    
    # Show most significant pathways
    if summary['most_significant_pathways']:
        print(f"\nTop significant pathways:")
        for i, pathway in enumerate(summary['most_significant_pathways'][:5]):
            if 'pathway_name' in pathway:
                print(f"  {i+1}. {pathway['pathway_name']} (p={pathway.get('p_value_corrected', pathway['p_value']):.2e})")
            elif 'go_name' in pathway:
                print(f"  {i+1}. {pathway['go_name']} [{pathway['go_category']}] (p={pathway.get('p_value_corrected', pathway['p_value']):.2e})")
    
    # Test different parameter configurations
    print(f"\n🔧 Testing different parameter configurations...")
    
    # Test KEGG-only
    kegg_params = PathwayEnrichmentParams(use_go=False, use_kegg=True)
    kegg_results = run_pathway_enrichment(feature_df, significant_features, params=kegg_params)
    print(f"  KEGG-only: {kegg_results['summary']['total_significant_pathways']} pathways")
    
    # Test GO-only  
    go_params = PathwayEnrichmentParams(use_kegg=False, use_go=True)
    go_results = run_pathway_enrichment(feature_df, significant_features, params=go_params)
    print(f"  GO-only: {go_results['summary']['total_significant_pathways']} terms")
    
    # Test stricter p-value threshold
    strict_params = PathwayEnrichmentParams(p_value_threshold=0.01)
    strict_results = run_pathway_enrichment(feature_df, significant_features, params=strict_params)
    print(f"  Strict p<0.01: {strict_results['summary']['total_significant_pathways']} pathways")
    
    # Save results
    print(f"\n💾 Saving results...")
    output_dir = Path("enrichment_test_output")
    
    from biomarker.core.analysis.pathway_analysis import save_enrichment_results
    saved_files = save_enrichment_results(results, output_dir)
    
    print(f"Results saved to: {output_dir}")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")
    
    print(f"\n✅ Pathway enrichment analysis test completed!")
    print(f"🧬 Biological interpretation is now available for biomarker results!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pathway Enrichment Analysis Test")
    parser.add_argument("--p_threshold", type=float, default=0.05, help="p-value threshold for significance")
    parser.add_argument("--fdr_method", type=str, default="fdr_bh", help="FDR method to use")
    parser.add_argument("--no_kegg", action="store_true", help="Exclude KEGG pathways")
    parser.add_argument("--no_go", action="store_true", help="Exclude GO terms")
    parser.add_argument("--organism", type=str, help="Organism for pathway analysis")
    args = parser.parse_args()

    # Set parameters
    params = PathwayEnrichmentParams(
        p_value_threshold=args.p_threshold,
        fdr_method=args.fdr_method,
        use_kegg=not args.no_kegg,
        use_go=not args.no_go,
        organism=args.organism
    )

    main() 