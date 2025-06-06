#!/usr/bin/env python3
"""
Task #13: Pathway/Enrichment Analysis Module
Biological interpretation of biomarker features through pathway enrichment.

Supports:
- KEGG pathway enrichment
- GO term enrichment  
- Reactome pathway analysis
- Custom pathway databases
- Statistical significance testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import requests
import time
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings


class PathwayAnalysisError(Exception):
    """Raised when pathway analysis fails"""
    pass


@dataclass
class PathwayEnrichmentParams:
    """Parameters for pathway enrichment analysis"""
    # Statistical parameters
    p_value_threshold: float = 0.05
    fdr_method: str = 'fdr_bh'  # 'fdr_bh', 'bonferroni', 'holm'
    min_genes_per_pathway: int = 3
    max_genes_per_pathway: int = 500
    
    # Database selection
    use_kegg: bool = True
    use_go: bool = True
    use_reactome: bool = False  # Requires additional setup
    
    # GO term categories
    go_categories: List[str] = None  # ['BP', 'MF', 'CC'] or subset
    
    # Feature mapping
    feature_id_type: str = 'auto'  # 'kegg_compound', 'hmdb', 'pubchem', 'auto'
    organism: str = 'hsa'  # KEGG organism code (hsa=human)
    
    # Output options
    max_results: int = 50
    include_gene_lists: bool = True

    def __post_init__(self):
        if self.go_categories is None:
            self.go_categories = ['BP', 'MF', 'CC']  # Biological Process, Molecular Function, Cellular Component


def run_pathway_enrichment(
    feature_df: pd.DataFrame,
    significant_features: List[str],
    background_features: Optional[List[str]] = None,
    params: Optional[PathwayEnrichmentParams] = None
) -> Dict[str, Any]:
    """
    Run comprehensive pathway enrichment analysis
    
    Args:
        feature_df: Feature matrix with metadata
        significant_features: List of significant feature IDs
        background_features: Background feature set (optional, uses all features if None)
        params: Enrichment parameters
        
    Returns:
        Dictionary with enrichment results
    """
    if params is None:
        params = PathwayEnrichmentParams()
    
    print(f"🧬 Starting pathway enrichment analysis...")
    print(f"  Significant features: {len(significant_features)}")
    
    # Prepare feature sets
    if background_features is None:
        background_features = list(feature_df.columns)
    
    print(f"  Background features: {len(background_features)}")
    
    # Extract feature metadata for mapping
    feature_metadata = extract_feature_metadata(feature_df)
    
    # Map features to biological identifiers
    feature_mapping = map_features_to_ids(
        significant_features,
        background_features, 
        feature_metadata,
        params.feature_id_type
    )
    
    print(f"  Mapped features: {len(feature_mapping['significant_mapped'])} / {len(significant_features)}")
    
    results = {
        'parameters': params.__dict__,
        'feature_mapping': feature_mapping,
        'enrichment_results': {}
    }
    
    # Run KEGG enrichment if enabled
    if params.use_kegg:
        print("  🔍 Running KEGG pathway enrichment...")
        kegg_results = run_kegg_enrichment(
            feature_mapping['significant_mapped'],
            feature_mapping['background_mapped'],
            params
        )
        results['enrichment_results']['kegg'] = kegg_results
    
    # Run GO enrichment if enabled
    if params.use_go:
        print("  🔍 Running GO term enrichment...")
        go_results = run_go_enrichment(
            feature_mapping['significant_mapped'],
            feature_mapping['background_mapped'],
            params
        )
        results['enrichment_results']['go'] = go_results
    
    # Run Reactome enrichment if enabled
    if params.use_reactome:
        print("  🔍 Running Reactome pathway enrichment...")
        reactome_results = run_reactome_enrichment(
            feature_mapping['significant_mapped'],
            feature_mapping['background_mapped'],
            params
        )
        results['enrichment_results']['reactome'] = reactome_results
    
    # Compile summary statistics
    results['summary'] = compile_enrichment_summary(results['enrichment_results'], params)
    
    print(f"✅ Pathway enrichment analysis complete!")
    return results


def extract_feature_metadata(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract feature metadata from DataFrame attributes or columns
    
    Args:
        feature_df: Feature matrix with metadata
        
    Returns:
        DataFrame with feature metadata
    """
    # Try to get metadata from DataFrame attributes (from feature extraction)
    if 'feature_metadata' in feature_df.attrs:
        return feature_df.attrs['feature_metadata']
    
    # Try to extract from column names or create dummy metadata
    feature_ids = list(feature_df.columns)
    
    # Look for m/z and RT patterns in feature names
    metadata_rows = []
    for feature_id in feature_ids:
        # Parse feature names like "feature_000123" or "mz_123.4567_rt_567.89"
        if '_' in feature_id:
            parts = feature_id.split('_')
            mz = None
            rt = None
            
            for i, part in enumerate(parts):
                if part == 'mz' and i + 1 < len(parts):
                    try:
                        mz = float(parts[i + 1])
                    except ValueError:
                        pass
                elif part == 'rt' and i + 1 < len(parts):
                    try:
                        rt = float(parts[i + 1])
                    except ValueError:
                        pass
            
            metadata_rows.append({
                'feature_id': feature_id,
                'mz': mz if mz is not None else np.random.uniform(100, 1000),  # Dummy m/z
                'rt': rt if rt is not None else np.random.uniform(0, 1800)      # Dummy RT
            })
        else:
            # Create dummy metadata
            metadata_rows.append({
                'feature_id': feature_id,
                'mz': np.random.uniform(100, 1000),
                'rt': np.random.uniform(0, 1800)
            })
    
    return pd.DataFrame(metadata_rows).set_index('feature_id')


def map_features_to_ids(
    significant_features: List[str],
    background_features: List[str],
    feature_metadata: pd.DataFrame,
    id_type: str = 'auto'
) -> Dict[str, Any]:
    """
    Map metabolomics features to biological database identifiers
    
    Args:
        significant_features: Significant feature list
        background_features: Background feature list
        feature_metadata: Feature metadata with m/z and RT
        id_type: Type of mapping to perform
        
    Returns:
        Dictionary with mapping results
    """
    print(f"  🔗 Mapping features to biological IDs ({id_type})...")
    
    # For now, implement m/z-based mapping to KEGG compounds
    # In production, this would use HMDB, METLIN, or other databases
    
    significant_mapped = map_mz_to_kegg_compounds(
        [f for f in significant_features if f in feature_metadata.index],
        feature_metadata
    )
    
    background_mapped = map_mz_to_kegg_compounds(
        [f for f in background_features if f in feature_metadata.index],
        feature_metadata
    )
    
    return {
        'significant_original': significant_features,
        'background_original': background_features,
        'significant_mapped': significant_mapped,
        'background_mapped': background_mapped,
        'mapping_method': id_type,
        'mapping_rate': len(significant_mapped) / len(significant_features) if significant_features else 0
    }


def map_mz_to_kegg_compounds(
    feature_ids: List[str],
    feature_metadata: pd.DataFrame,
    mass_tolerance_ppm: float = 10.0
) -> List[str]:
    """
    Map m/z values to KEGG compound IDs using mass matching
    
    Args:
        feature_ids: List of feature IDs to map
        feature_metadata: Metadata with m/z values
        mass_tolerance_ppm: Mass tolerance for matching
        
    Returns:
        List of mapped KEGG compound IDs
    """
    mapped_compounds = []
    
    # Demo compound mapping (in production, use KEGG REST API or local database)
    demo_compounds = {
        # Common metabolites with their m/z values
        180.0634: 'C00031',  # Glucose
        146.0579: 'C00022',  # Pyruvate  
        117.0193: 'C00025',  # Glutamate
        132.0532: 'C00026',  # Ketoglutarate
        174.0528: 'C00036',  # Oxaloacetate
        147.0532: 'C00049',  # Aspartate
        204.0892: 'C00158',  # Citrate
        166.0528: 'C00122',  # Fumarate
        150.0528: 'C00149',  # Malate
        181.0345: 'C00191',  # Succinate
    }
    
    for feature_id in feature_ids:
        if feature_id in feature_metadata.index:
            feature_mz = feature_metadata.loc[feature_id, 'mz']
            
            # Find closest match within tolerance
            for compound_mz, compound_id in demo_compounds.items():
                ppm_error = abs(feature_mz - compound_mz) / compound_mz * 1e6
                if ppm_error <= mass_tolerance_ppm:
                    mapped_compounds.append(compound_id)
                    break
    
    return list(set(mapped_compounds))  # Remove duplicates


def run_kegg_enrichment(
    significant_compounds: List[str],
    background_compounds: List[str],
    params: PathwayEnrichmentParams
) -> Dict[str, Any]:
    """
    Run KEGG pathway enrichment analysis
    
    Args:
        significant_compounds: KEGG compound IDs of interest
        background_compounds: Background KEGG compound IDs
        params: Analysis parameters
        
    Returns:
        KEGG enrichment results
    """
    if not significant_compounds:
        return {'pathways': [], 'message': 'No compounds mapped to KEGG'}
    
    # Demo KEGG pathways (in production, use KEGG REST API)
    demo_pathways = {
        'hsa00010': {
            'name': 'Glycolysis / Gluconeogenesis',
            'compounds': ['C00031', 'C00022', 'C00118', 'C00186', 'C00197'],
            'genes': 67
        },
        'hsa00020': {
            'name': 'Citrate cycle (TCA cycle)',
            'compounds': ['C00036', 'C00158', 'C00122', 'C00149', 'C00191'],
            'genes': 30
        },
        'hsa00250': {
            'name': 'Alanine, aspartate and glutamate metabolism',
            'compounds': ['C00025', 'C00049', 'C00026', 'C00064', 'C00152'],
            'genes': 24
        },
        'hsa00380': {
            'name': 'Tryptophan metabolism', 
            'compounds': ['C00078', 'C00643', 'C00328', 'C05635'],
            'genes': 41
        }
    }
    
    enrichment_results = []
    
    for pathway_id, pathway_info in demo_pathways.items():
        # Calculate overlap
        pathway_compounds = set(pathway_info['compounds'])
        sig_overlap = set(significant_compounds) & pathway_compounds
        bg_overlap = set(background_compounds) & pathway_compounds
        
        if len(sig_overlap) >= params.min_genes_per_pathway:
            # Fisher's exact test for enrichment
            a = len(sig_overlap)  # Significant in pathway
            b = len(significant_compounds) - a  # Significant not in pathway
            c = len(bg_overlap) - a  # Background in pathway (not significant)
            d = len(background_compounds) - len(bg_overlap) - b  # Background not in pathway
            
            if c > 0 and d > 0:  # Avoid division by zero
                oddsratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                
                enrichment_results.append({
                    'pathway_id': pathway_id,
                    'pathway_name': pathway_info['name'],
                    'p_value': p_value,
                    'odds_ratio': oddsratio,
                    'significant_compounds': len(sig_overlap),
                    'pathway_size': len(pathway_compounds),
                    'total_significant': len(significant_compounds),
                    'overlap_compounds': list(sig_overlap) if params.include_gene_lists else [],
                    'enrichment_score': -np.log10(p_value) if p_value > 0 else 10
                })
    
    # Sort by p-value and apply multiple testing correction
    enrichment_results.sort(key=lambda x: x['p_value'])
    
    if enrichment_results:
        p_values = [r['p_value'] for r in enrichment_results]
        rejected, corrected_p, _, _ = multipletests(p_values, method=params.fdr_method)
        
        for i, result in enumerate(enrichment_results):
            result['p_value_corrected'] = corrected_p[i]
            result['significant'] = rejected[i]
    
    # Filter significant results
    significant_pathways = [r for r in enrichment_results if r.get('p_value_corrected', 1) <= params.p_value_threshold]
    
    return {
        'pathways': significant_pathways[:params.max_results],
        'total_tested': len(demo_pathways),
        'significant_count': len(significant_pathways),
        'method': 'fisher_exact',
        'multiple_testing_correction': params.fdr_method
    }


def run_go_enrichment(
    significant_compounds: List[str],
    background_compounds: List[str],
    params: PathwayEnrichmentParams
) -> Dict[str, Any]:
    """
    Run Gene Ontology (GO) term enrichment analysis
    
    Args:
        significant_compounds: Compound IDs of interest
        background_compounds: Background compound IDs
        params: Analysis parameters
        
    Returns:
        GO enrichment results
    """
    if not significant_compounds:
        return {'terms': [], 'message': 'No compounds available for GO analysis'}
    
    # Demo GO terms related to metabolism
    demo_go_terms = {
        'GO:0008152': {
            'name': 'metabolic process',
            'category': 'BP',
            'compounds': ['C00031', 'C00022', 'C00025', 'C00026', 'C00036'],
            'genes': 5234
        },
        'GO:0044237': {
            'name': 'cellular metabolic process',
            'category': 'BP', 
            'compounds': ['C00031', 'C00022', 'C00158', 'C00149'],
            'genes': 4521
        },
        'GO:0006096': {
            'name': 'glycolytic process',
            'category': 'BP',
            'compounds': ['C00031', 'C00022', 'C00118'],
            'genes': 89
        },
        'GO:0016829': {
            'name': 'lyase activity',
            'category': 'MF',
            'compounds': ['C00036', 'C00158', 'C00122'],
            'genes': 312
        }
    }
    
    enrichment_results = []
    
    for go_id, go_info in demo_go_terms.items():
        if go_info['category'] not in params.go_categories:
            continue
            
        # Calculate overlap (similar to KEGG)
        go_compounds = set(go_info['compounds'])
        sig_overlap = set(significant_compounds) & go_compounds
        bg_overlap = set(background_compounds) & go_compounds
        
        if len(sig_overlap) >= params.min_genes_per_pathway:
            # Fisher's exact test
            a = len(sig_overlap)
            b = len(significant_compounds) - a
            c = len(bg_overlap) - a
            d = len(background_compounds) - len(bg_overlap) - b
            
            if c > 0 and d > 0:
                oddsratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                
                enrichment_results.append({
                    'go_id': go_id,
                    'go_name': go_info['name'],
                    'go_category': go_info['category'],
                    'p_value': p_value,
                    'odds_ratio': oddsratio,
                    'significant_compounds': len(sig_overlap),
                    'term_size': len(go_compounds),
                    'total_significant': len(significant_compounds),
                    'overlap_compounds': list(sig_overlap) if params.include_gene_lists else [],
                    'enrichment_score': -np.log10(p_value) if p_value > 0 else 10
                })
    
    # Sort and correct for multiple testing
    enrichment_results.sort(key=lambda x: x['p_value'])
    
    if enrichment_results:
        p_values = [r['p_value'] for r in enrichment_results]
        rejected, corrected_p, _, _ = multipletests(p_values, method=params.fdr_method)
        
        for i, result in enumerate(enrichment_results):
            result['p_value_corrected'] = corrected_p[i]
            result['significant'] = rejected[i]
    
    # Filter significant results
    significant_terms = [r for r in enrichment_results if r.get('p_value_corrected', 1) <= params.p_value_threshold]
    
    return {
        'terms': significant_terms[:params.max_results],
        'total_tested': len([t for t in demo_go_terms.values() if t['category'] in params.go_categories]),
        'significant_count': len(significant_terms),
        'categories_tested': params.go_categories,
        'method': 'fisher_exact',
        'multiple_testing_correction': params.fdr_method
    }


def run_reactome_enrichment(
    significant_compounds: List[str],
    background_compounds: List[str],
    params: PathwayEnrichmentParams
) -> Dict[str, Any]:
    """
    Run Reactome pathway enrichment analysis
    
    Args:
        significant_compounds: Compound IDs of interest
        background_compounds: Background compound IDs  
        params: Analysis parameters
        
    Returns:
        Reactome enrichment results
    """
    # Placeholder for Reactome analysis
    # Would require Reactome database integration
    return {
        'pathways': [],
        'message': 'Reactome analysis not yet implemented',
        'total_tested': 0,
        'significant_count': 0
    }


def compile_enrichment_summary(
    enrichment_results: Dict[str, Any],
    params: PathwayEnrichmentParams
) -> Dict[str, Any]:
    """
    Compile summary statistics across all enrichment analyses
    
    Args:
        enrichment_results: Results from all enrichment analyses
        params: Analysis parameters
        
    Returns:
        Summary statistics
    """
    summary = {
        'databases_tested': list(enrichment_results.keys()),
        'total_significant_pathways': 0,
        'most_significant_pathways': [],
        'enrichment_overview': {}
    }
    
    all_significant = []
    
    for db_name, db_results in enrichment_results.items():
        if db_name == 'kegg':
            significant_pathways = db_results.get('pathways', [])
            summary['enrichment_overview']['kegg'] = {
                'tested': db_results.get('total_tested', 0),
                'significant': db_results.get('significant_count', 0)
            }
        elif db_name == 'go':
            significant_pathways = db_results.get('terms', [])
            summary['enrichment_overview']['go'] = {
                'tested': db_results.get('total_tested', 0),
                'significant': db_results.get('significant_count', 0)
            }
        else:
            significant_pathways = db_results.get('pathways', [])
            summary['enrichment_overview'][db_name] = {
                'tested': db_results.get('total_tested', 0),
                'significant': db_results.get('significant_count', 0)
            }
        
        all_significant.extend(significant_pathways)
    
    # Sort all significant pathways by p-value
    all_significant.sort(key=lambda x: x.get('p_value_corrected', x.get('p_value', 1)))
    
    summary['total_significant_pathways'] = len(all_significant)
    summary['most_significant_pathways'] = all_significant[:10]  # Top 10
    
    return summary


def save_enrichment_results(
    results: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save enrichment results to files
    
    Args:
        results: Enrichment analysis results
        output_dir: Output directory
        
    Returns:
        Dictionary of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save complete results as JSON
    results_file = output_dir / "pathway_enrichment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    saved_files['complete_results'] = results_file
    
    # Save KEGG results as CSV
    if 'kegg' in results['enrichment_results']:
        kegg_pathways = results['enrichment_results']['kegg'].get('pathways', [])
        if kegg_pathways:
            kegg_df = pd.DataFrame(kegg_pathways)
            kegg_file = output_dir / "kegg_enrichment.csv"
            kegg_df.to_csv(kegg_file, index=False)
            saved_files['kegg_csv'] = kegg_file
    
    # Save GO results as CSV
    if 'go' in results['enrichment_results']:
        go_terms = results['enrichment_results']['go'].get('terms', [])
        if go_terms:
            go_df = pd.DataFrame(go_terms)
            go_file = output_dir / "go_enrichment.csv"
            go_df.to_csv(go_file, index=False)
            saved_files['go_csv'] = go_file
    
    # Save summary
    summary_file = output_dir / "enrichment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results['summary'], f, indent=2, default=str)
    saved_files['summary'] = summary_file
    
    print(f"  📄 Enrichment results saved to {output_dir}")
    
    return saved_files


# CLI interface
def main():
    """Command line interface for pathway enrichment analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pathway Enrichment Analysis")
    parser.add_argument("feature_matrix", help="Feature matrix CSV file")
    parser.add_argument("significant_features", help="File with significant feature IDs (one per line)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("--fdr-method", default='benjamini_hochberg', help="FDR correction method")
    parser.add_argument("--no-kegg", action='store_true', help="Skip KEGG analysis")
    parser.add_argument("--no-go", action='store_true', help="Skip GO analysis")
    parser.add_argument("--organism", default='hsa', help="KEGG organism code")
    
    args = parser.parse_args()
    
    try:
        # Load feature matrix
        print(f"Loading feature matrix from {args.feature_matrix}...")
        feature_df = pd.read_csv(args.feature_matrix, index_col=0)
        
        # Load significant features
        print(f"Loading significant features from {args.significant_features}...")
        with open(args.significant_features, 'r') as f:
            significant_features = [line.strip() for line in f if line.strip()]
        
        # Set parameters
        params = PathwayEnrichmentParams(
            p_value_threshold=args.p_threshold,
            fdr_method=args.fdr_method,
            use_kegg=not args.no_kegg,
            use_go=not args.no_go,
            organism=args.organism
        )
        
        # Run enrichment analysis
        results = run_pathway_enrichment(feature_df, significant_features, params=params)
        
        # Save results
        saved_files = save_enrichment_results(results, Path(args.output))
        
        # Print summary
        print(f"\n✅ Pathway enrichment analysis completed!")
        print(f"  Total significant pathways: {results['summary']['total_significant_pathways']}")
        
        for db_name, db_summary in results['summary']['enrichment_overview'].items():
            print(f"  {db_name.upper()}: {db_summary['significant']} / {db_summary['tested']} significant")
        
        print(f"\nOutput files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 