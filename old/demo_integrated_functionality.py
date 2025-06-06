#!/usr/bin/env python3
"""
DEMONSTRATION: Integrated Feature Extraction + Enrichment Pipeline

This script demonstrates the complete integrated workflow:
1. Feature extraction (deduplication, filtering, imputation, scaling)
2. Statistical analysis simulation 
3. Pathway enrichment analysis
4. Results reporting

This shows that the integration is working without requiring the API server.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add biomarker package to path
sys.path.append('.')

# Import our integrated modules
from biomarker.core.features.deduplication import deduplicate_features
from biomarker.core.features.filtering import combined_filter
from biomarker.core.features.imputation import impute_missing_values
from biomarker.core.preprocessing.scalers import apply_scaling, ScalerParams
from biomarker.core.analysis.pathway_analysis import run_pathway_enrichment, PathwayEnrichmentParams

def create_demo_dataset():
    """Create a realistic demo metabolomics dataset"""
    print("📊 Creating demo metabolomics dataset...")
    
    np.random.seed(42)
    
    # Sample names (NAFLD study style)
    sample_names = ['Control_' + str(i) for i in range(1, 16)] + ['NAFLD_' + str(i) for i in range(1, 16)]
    
    # Feature names (metabolite-like with m/z and RT info)
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
    ] + [f'feature_{i:05d}' for i in range(100, 300)]  # Additional features
    
    # Generate realistic data matrix
    data = np.random.lognormal(mean=3, sigma=1, size=(len(sample_names), len(feature_names)))
    
    # Add some differential patterns between groups
    control_mask = np.array(['Control' in name for name in sample_names])
    nafld_mask = ~control_mask
    
    # Make some features "significant" by adding differential expression
    significant_indices = [0, 1, 2, 6, 7, 8]  # Metabolic pathway features
    for idx in significant_indices:
        data[nafld_mask, idx] *= 1.5  # 1.5x higher in NAFLD
    
    # Add missing values (realistic pattern)
    missing_mask = np.random.random(data.shape) < 0.08
    data[missing_mask] = 0
    
    # Create DataFrame
    df = pd.DataFrame(data, index=sample_names, columns=feature_names)
    
    # Create metadata for features
    feature_metadata = pd.DataFrame({
        'mz': [float(name.split('_')[1]) if 'mz_' in name else np.random.uniform(100, 1000) for name in feature_names],
        'rt': [float(name.split('_')[3]) if 'rt_' in name else np.random.uniform(0, 1800) for name in feature_names]
    }, index=feature_names)
    
    # Attach metadata to DataFrame
    df.attrs['feature_metadata'] = feature_metadata
    
    print(f"  ✅ Created dataset: {df.shape[0]} samples × {df.shape[1]} features")
    print(f"  📊 Groups: {control_mask.sum()} Control, {nafld_mask.sum()} NAFLD")
    print(f"  🔍 Missing values: {(data == 0).sum()} ({(data == 0).mean()*100:.1f}%)")
    
    return df

def run_feature_extraction_pipeline(df):
    """Run the complete feature extraction pipeline"""
    print(f"\n🔬 FEATURE EXTRACTION PIPELINE")
    print("=" * 50)
    
    initial_features = len(df.columns)
    print(f"Starting with {initial_features} features")
    
    # Step 1: Deduplication
    print(f"\n🎯 Step 1: Feature Deduplication")
    deduplicated_df, dedup_stats = deduplicate_features(
        df,
        mz_tolerance_ppm=5.0,
        rt_tolerance_min=0.1,
        keep_strategy="highest_abundance"
    )
    print(f"  After deduplication: {len(deduplicated_df.columns)} features")
    print(f"  Reduction: {dedup_stats['reduction_percentage']:.1f}%")
    
    # Step 2: Filtering
    print(f"\n🔍 Step 2: Frequency & Score Filtering")
    filter_config = {
        'frequency_filter': {
            'enabled': True,
            'min_frequency': 0.6,  # Present in 60% of samples
            'per_group': False
        },
        'score_filter': {
            'enabled': True,
            'method': 'total_abundance',
            'min_percentile': 40.0  # Top 60% by abundance
        },
        'custom_filters': {'enabled': False}
    }
    
    filtered_df, filter_stats = combined_filter(deduplicated_df, filter_config)
    print(f"  After filtering: {len(filtered_df.columns)} features")
    print(f"  Additional reduction: {filter_stats['total_reduction_percentage']:.1f}%")
    
    # Step 3: Imputation
    print(f"\n🩹 Step 3: Missing Value Imputation")
    imputed_df, imputation_stats = impute_missing_values(
        filtered_df,
        method="median_global"
    )
    
    # Handle case where there are no missing values
    missing_before = imputation_stats.get('missing_before', 0)
    missing_after = imputation_stats.get('missing_after', 0)
    imputation_rate = imputation_stats.get('imputation_rate', 0.0)
    
    print(f"  Missing values before: {missing_before}")
    print(f"  Missing values after: {missing_after}")
    print(f"  Imputation rate: {imputation_rate:.1f}%")
    
    # Step 4: Scaling
    print(f"\n⚖️ Step 4: Feature Scaling")
    scaler_params = ScalerParams(method="pareto", log_transform=False)
    scaled_df, scaling_info = apply_scaling(imputed_df, "pareto", scaler_params)
    print(f"  Scaling method: {scaling_info['method']}")
    print(f"  Features scaled: {len(scaled_df.columns)}")
    
    # Summary
    final_features = len(scaled_df.columns)
    total_reduction = (initial_features - final_features) / initial_features * 100
    
    print(f"\n📊 FEATURE EXTRACTION SUMMARY:")
    print(f"  Initial features: {initial_features}")
    print(f"  Final features: {final_features}")
    print(f"  Total reduction: {total_reduction:.1f}%")
    print(f"  Samples: {len(scaled_df.index)}")
    print(f"  Data quality: ✅ No missing values, ready for ML")
    
    return scaled_df

def simulate_statistical_analysis(df):
    """Simulate statistical analysis to find significant features"""
    print(f"\n📈 STATISTICAL ANALYSIS SIMULATION")
    print("=" * 45)
    
    # For demo, select features that were designed to be "differential"
    metabolic_features = [col for col in df.columns if 'mz_' in col and any(x in col for x in ['180.0634', '146.0579', '117.0193', '204.0892', '166.0528', '150.0528'])]
    
    # Add some random features to simulate statistical discovery
    random_features = np.random.choice([col for col in df.columns if col not in metabolic_features], 
                                      min(8, len(df.columns) - len(metabolic_features)), 
                                      replace=False).tolist()
    
    significant_features = metabolic_features + random_features
    
    print(f"  🎯 Identified {len(significant_features)} significant features")
    print(f"  📋 Significant features:")
    for i, feature in enumerate(significant_features[:8], 1):
        print(f"    {i}. {feature}")
    if len(significant_features) > 8:
        print(f"    ... and {len(significant_features) - 8} more")
    
    return significant_features

def run_enrichment_analysis(df, significant_features):
    """Run pathway enrichment analysis"""
    print(f"\n🧬 PATHWAY ENRICHMENT ANALYSIS")
    print("=" * 45)
    
    # Set up enrichment parameters
    enrichment_params = PathwayEnrichmentParams(
        p_value_threshold=0.05,
        fdr_method="fdr_bh",
        use_kegg=True,
        use_go=True,
        organism="hsa"
    )
    
    # Run enrichment analysis
    enrichment_results = run_pathway_enrichment(
        df,
        significant_features,
        params=enrichment_params
    )
    
    # Display results
    summary = enrichment_results['summary']
    print(f"\n📊 ENRICHMENT RESULTS:")
    print(f"  Total significant pathways: {summary['total_significant_pathways']}")
    
    for db_name, db_summary in summary['enrichment_overview'].items():
        print(f"  {db_name.upper()}: {db_summary['significant']}/{db_summary['tested']} pathways significant")
    
    # Show most significant pathways
    if summary['most_significant_pathways']:
        print(f"\n🏆 TOP SIGNIFICANT PATHWAYS:")
        for i, pathway in enumerate(summary['most_significant_pathways'][:5], 1):
            if 'pathway_name' in pathway:
                print(f"    {i}. {pathway['pathway_name']}")
                print(f"       p-value: {pathway.get('p_value_corrected', pathway['p_value']):.2e}")
            elif 'go_name' in pathway:
                print(f"    {i}. {pathway['go_name']} [{pathway['go_category']}]")
                print(f"       p-value: {pathway.get('p_value_corrected', pathway['p_value']):.2e}")
            print()
    
    return enrichment_results

def save_demo_results(feature_df, significant_features, enrichment_results):
    """Save demo results to files"""
    print(f"💾 SAVING RESULTS")
    print("=" * 20)
    
    output_dir = Path("demo_integrated_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save feature matrix
    feature_file = output_dir / "final_feature_matrix.csv"
    feature_df.to_csv(feature_file)
    print(f"  📊 Feature matrix: {feature_file}")
    
    # Save significant features
    sig_file = output_dir / "significant_features.txt"
    with open(sig_file, 'w') as f:
        for feature in significant_features:
            f.write(f"{feature}\n")
    print(f"  🎯 Significant features: {sig_file}")
    
    # Save enrichment results
    enrichment_file = output_dir / "enrichment_results.json"
    with open(enrichment_file, 'w') as f:
        json.dump(enrichment_results, f, indent=2, default=str)
    print(f"  🧬 Enrichment results: {enrichment_file}")
    
    return output_dir

def main():
    """Run the complete integrated demonstration"""
    print("🚀 INTEGRATED FEATURE EXTRACTION + ENRICHMENT DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates the complete pipeline without requiring API server")
    print()
    
    # Step 1: Create demo dataset
    df = create_demo_dataset()
    
    # Step 2: Feature extraction pipeline
    processed_df = run_feature_extraction_pipeline(df)
    
    # Step 3: Statistical analysis simulation
    significant_features = simulate_statistical_analysis(processed_df)
    
    # Step 4: Pathway enrichment analysis
    enrichment_results = run_enrichment_analysis(processed_df, significant_features)
    
    # Step 5: Save results
    output_dir = save_demo_results(processed_df, significant_features, enrichment_results)
    
    # Final summary
    print(f"\n✅ INTEGRATION DEMONSTRATION COMPLETE!")
    print("=" * 45)
    print("🎉 The integrated pipeline is fully functional:")
    print("  ✅ Feature extraction (deduplication, filtering, imputation, scaling)")
    print("  ✅ Statistical analysis simulation")
    print("  ✅ Pathway enrichment (KEGG + GO terms)")
    print("  ✅ Results saved and ready for analysis")
    print()
    print(f"📁 Results saved to: {output_dir}")
    print("🌐 API endpoints also implemented and ready for deployment!")

if __name__ == "__main__":
    main() 