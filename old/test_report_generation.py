#!/usr/bin/env python3
"""
Test Script: Comprehensive Report Generation

This script demonstrates the report generation capabilities
using the results from our integrated feature extraction + enrichment pipeline.
"""

import sys
import json
from pathlib import Path

# Add biomarker package to path
sys.path.append('.')

# Import report generator
from biomarker.report.report_generator import (
    generate_comprehensive_report,
    generate_executive_summary,
    ReportConfig
)

def create_demo_analysis_results():
    """Create demo analysis results for report generation"""
    
    # Simulate comprehensive analysis results
    analysis_results = {
        'feature_extraction': {
            'initial_features': 210,
            'final_features': 56,
            'samples': 30,
            'reduction_percentage': 73.3,
            'pipeline_steps': [
                'Peak Detection',
                'Feature Deduplication',
                'Frequency Filtering',
                'Score Filtering',
                'Missing Value Imputation',
                'Pareto Scaling'
            ],
            'quality_metrics': {
                'missing_values_before': 15,
                'missing_values_after': 0,
                'feature_completeness': 100.0,
                'sample_coverage': 100.0
            }
        },
        'statistical_analysis': {
            'significant_features': [
                'mz_180.0634_rt_120.5',
                'mz_146.0579_rt_85.2',
                'mz_117.0193_rt_95.8',
                'mz_204.0892_rt_105.6',
                'mz_166.0528_rt_78.9',
                'feature_0125',
                'feature_0087',
                'feature_0156'
            ],
            'total_features_tested': 56,
            'p_value_threshold': 0.05,
            'correction_method': 'Benjamini-Hochberg FDR',
            'top_features': [
                {
                    'feature_id': 'mz_180.0634_rt_120.5',
                    'p_value': 0.001234,
                    'effect_size': 1.45,
                    'fold_change': 2.3
                },
                {
                    'feature_id': 'mz_146.0579_rt_85.2',
                    'p_value': 0.002456,
                    'effect_size': 1.21,
                    'fold_change': 1.8
                },
                {
                    'feature_id': 'mz_117.0193_rt_95.8',
                    'p_value': 0.003789,
                    'effect_size': 0.98,
                    'fold_change': 1.6
                },
                {
                    'feature_id': 'mz_204.0892_rt_105.6',
                    'p_value': 0.004521,
                    'effect_size': 0.87,
                    'fold_change': 1.4
                },
                {
                    'feature_id': 'mz_166.0528_rt_78.9',
                    'p_value': 0.006234,
                    'effect_size': 0.75,
                    'fold_change': 1.3
                }
            ],
            'effect_sizes': {
                'large_effect_count': 2,
                'medium_effect_count': 3,
                'small_effect_count': 3
            }
        },
        'pathway_analysis': {
            'summary': {
                'total_significant_pathways': 12,
                'enrichment_overview': {
                    'kegg': {
                        'significant': 7,
                        'tested': 145,
                        'enrichment_rate': 4.8
                    },
                    'go_bp': {
                        'significant': 3,
                        'tested': 89,
                        'enrichment_rate': 3.4
                    },
                    'go_mf': {
                        'significant': 2,
                        'tested': 67,
                        'enrichment_rate': 3.0
                    }
                },
                'most_significant_pathways': [
                    {
                        'pathway_name': 'Glycolysis / Gluconeogenesis',
                        'pathway_id': 'hsa00010',
                        'p_value': 0.0001234,
                        'p_value_corrected': 0.0012456,
                        'genes_in_pathway': 8,
                        'database': 'KEGG'
                    },
                    {
                        'pathway_name': 'Citrate cycle (TCA cycle)',
                        'pathway_id': 'hsa00020',
                        'p_value': 0.0002345,
                        'p_value_corrected': 0.0018567,
                        'genes_in_pathway': 6,
                        'database': 'KEGG'
                    },
                    {
                        'go_name': 'cellular carbohydrate metabolic process',
                        'go_id': 'GO:0044262',
                        'go_category': 'BP',
                        'p_value': 0.0003456,
                        'p_value_corrected': 0.0023456,
                        'genes_in_pathway': 5,
                        'database': 'GO'
                    },
                    {
                        'pathway_name': 'Fatty acid biosynthesis',
                        'pathway_id': 'hsa00061',
                        'p_value': 0.0004567,
                        'p_value_corrected': 0.0031234,
                        'genes_in_pathway': 4,
                        'database': 'KEGG'
                    },
                    {
                        'go_name': 'glucose metabolic process',
                        'go_id': 'GO:0006006',
                        'go_category': 'BP',
                        'p_value': 0.0005678,
                        'p_value_corrected': 0.0034567,
                        'genes_in_pathway': 4,
                        'database': 'GO'
                    }
                ]
            }
        },
        'quality_control': {
            'batch_effects': 'None detected',
            'outliers': 'None identified',
            'data_completeness': 100.0,
            'normalization_quality': 'Excellent'
        },
        'metadata': {
            'study_design': 'NAFLD vs Control comparison',
            'sample_groups': ['Control', 'NAFLD'],
            'analytical_platform': 'LC-MS/MS',
            'analysis_date': '2024-01-15',
            'total_analysis_time': '45 minutes'
        }
    }
    
    return analysis_results


def test_comprehensive_report():
    """Test comprehensive report generation"""
    print("📋 Testing Comprehensive Report Generation")
    print("=" * 50)
    
    # Create demo results
    analysis_results = create_demo_analysis_results()
    
    # Configure report for professional clinical setting
    config = ReportConfig(
        title="NAFLD Biomarker Discovery Analysis Report",
        project_name="NAFLD Metabolomics Study",
        author="Dr. Research Team",
        institution="University Medical Center",
        theme="professional",
        include_plots=False,  # Skip plots for demo
        technical_level="standard"
    )
    
    # Generate comprehensive report
    output_dir = Path("report_generation_test")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔬 Generating comprehensive report...")
    generated_files = generate_comprehensive_report(analysis_results, output_dir, config)
    
    print(f"\n✅ Report generation completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Generated files:")
    for file_type, file_path in generated_files.items():
        print(f"   {file_type.upper()}: {file_path}")
    
    return generated_files


def test_executive_summary():
    """Test executive summary generation"""
    print("\n📋 Testing Executive Summary Generation")
    print("=" * 45)
    
    # Create demo results
    analysis_results = create_demo_analysis_results()
    
    # Generate executive summary
    output_file = Path("report_generation_test") / "executive_summary.html"
    
    print(f"📊 Generating executive summary...")
    summary_file = generate_executive_summary(analysis_results, output_file)
    
    print(f"✅ Executive summary generated: {summary_file}")
    
    return summary_file


def test_clinical_theme():
    """Test clinical-themed report"""
    print("\n📋 Testing Clinical Theme Report")
    print("=" * 40)
    
    # Create demo results
    analysis_results = create_demo_analysis_results()
    
    # Configure for clinical audience
    config = ReportConfig(
        title="Clinical Biomarker Analysis Report",
        project_name="NAFLD Clinical Trial",
        author="Clinical Research Team",
        institution="Hospital Clinical Laboratory",
        theme="clinical",
        technical_level="basic",
        include_methodology=False,  # Less technical detail
        include_raw_data_summary=False
    )
    
    # Generate clinical report
    output_dir = Path("report_generation_test") / "clinical"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🏥 Generating clinical report...")
    generated_files = generate_comprehensive_report(analysis_results, output_dir, config)
    
    print(f"✅ Clinical report generated!")
    print(f"📄 Files: {list(generated_files.keys())}")
    
    return generated_files


def test_academic_theme():
    """Test academic-themed report"""
    print("\n📋 Testing Academic Theme Report")
    print("=" * 40)
    
    # Create demo results
    analysis_results = create_demo_analysis_results()
    
    # Configure for academic publication
    config = ReportConfig(
        title="Metabolomics Biomarker Discovery: A Systematic Analysis",
        project_name="NAFLD Metabolomics Research",
        author="Research Laboratory",
        institution="University Department of Biochemistry", 
        theme="academic",
        technical_level="detailed",
        include_plots=False,
        generate_pdf=False  # HTML only for demo
    )
    
    # Generate academic report
    output_dir = Path("report_generation_test") / "academic"
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎓 Generating academic report...")
    generated_files = generate_comprehensive_report(analysis_results, output_dir, config)
    
    print(f"✅ Academic report generated!")
    print(f"📄 Files: {list(generated_files.keys())}")
    
    return generated_files


def demonstrate_report_contents():
    """Demonstrate report contents by showing summary statistics"""
    print("\n📊 Report Contents Summary")
    print("=" * 30)
    
    # Load generated summary
    summary_file = Path("report_generation_test") / "report_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"📈 Analysis Summary:")
        fe_summary = summary['analysis_summary']['feature_extraction']
        print(f"   Initial Features: {fe_summary.get('initial_features', 'N/A')}")
        print(f"   Final Features: {fe_summary.get('final_features', 'N/A')}")
        print(f"   Feature Reduction: {fe_summary.get('reduction_percentage', 0):.1f}%")
        
        stat_summary = summary['analysis_summary']['statistical_analysis']
        print(f"   Significant Features: {stat_summary.get('significant_features_count', 'N/A')}")
        
        pathway_summary = summary['analysis_summary']['pathway_analysis']
        print(f"   Significant Pathways: {pathway_summary.get('total_significant_pathways', 'N/A')}")
        
        print(f"\n🕒 Generation Info:")
        gen_info = summary['generation_info']
        print(f"   Project: {gen_info.get('project_name', 'N/A')}")
        print(f"   Generated: {gen_info.get('generation_time', 'N/A')}")
    else:
        print("   ⚠️ Summary file not found")


def main():
    """Run comprehensive report generation tests"""
    print("🚀 COMPREHENSIVE REPORT GENERATION TEST")
    print("=" * 60)
    print("Testing biomarker discovery report generation capabilities")
    print()
    
    # Test 1: Comprehensive report
    generated_files = test_comprehensive_report()
    
    # Test 2: Executive summary
    summary_file = test_executive_summary()
    
    # Test 3: Clinical theme
    clinical_files = test_clinical_theme()
    
    # Test 4: Academic theme 
    academic_files = test_academic_theme()
    
    # Show contents summary
    demonstrate_report_contents()
    
    # Final summary
    print(f"\n✅ REPORT GENERATION TEST COMPLETE!")
    print("=" * 45)
    print("🎉 Successfully generated reports in multiple formats and themes:")
    print("  ✅ Professional HTML report with comprehensive sections")
    print("  ✅ Executive summary for clinical decision-making")
    print("  ✅ Clinical-themed report (simplified for clinicians)")
    print("  ✅ Academic-themed report (detailed for researchers)")
    print("  ✅ JSON summary statistics for programmatic access")
    print()
    print("📁 All reports saved to: report_generation_test/")
    print("🌐 Open the HTML files in a web browser to view reports")
    print("📤 Reports ready for sharing with stakeholders!")


if __name__ == "__main__":
    main() 