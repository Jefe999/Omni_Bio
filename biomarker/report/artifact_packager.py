#!/usr/bin/env python3
"""
Task #12: Artifact Packager
Packages all biomarker discovery results into organized, downloadable bundles.

This module collects outputs from:
- QC analysis (TIC/BPC plots, PCA)
- Statistical analysis (t-tests, volcano plots) 
- ML training (models, ROC curves, feature importance)
- Data processing (feature matrices, metadata)

And creates comprehensive packages with summary reports.
"""

import os
import shutil
import zipfile
import tarfile
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np


class ArtifactPackager:
    """
    Comprehensive artifact packager for biomarker discovery results.
    
    Creates organized, downloadable bundles with all analysis outputs,
    summary reports, and metadata.
    """
    
    def __init__(self, output_dir: str, project_name: str = "biomarker_discovery"):
        """
        Initialize the artifact packager.
        
        Args:
            output_dir: Base directory for package output
            project_name: Name for the analysis project
        """
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.package_name = f"{project_name}_{self.timestamp}"
        
        # Create package directory structure
        self.package_dir = self.output_dir / self.package_name
        self._create_directory_structure()
        
        # Track collected artifacts
        self.artifacts = {
            'qc_analysis': {},
            'statistical_analysis': {},
            'ml_models': {},
            'data_files': {},
            'metadata': {},
            'summary': {}
        }
        
    def _create_directory_structure(self):
        """Create organized directory structure for the package."""
        directories = [
            'qc_analysis',
            'qc_analysis/plots',
            'qc_analysis/data',
            'statistical_analysis', 
            'statistical_analysis/plots',
            'statistical_analysis/results',
            'ml_models',
            'ml_models/trained_models',
            'ml_models/evaluation',
            'ml_models/feature_importance',
            'data_files',
            'data_files/raw',
            'data_files/processed', 
            'data_files/metadata',
            'reports',
            'reports/html',
            'reports/json',
            'documentation'
        ]
        
        for directory in directories:
            (self.package_dir / directory).mkdir(parents=True, exist_ok=True)
            
    def add_qc_analysis(self, qc_dir: str, description: str = "Quality Control Analysis"):
        """
        Add QC analysis results to the package.
        
        Args:
            qc_dir: Directory containing QC outputs
            description: Description of the QC analysis
        """
        print(f"📊 Adding QC analysis from: {qc_dir}")
        
        qc_source = Path(qc_dir)
        if not qc_source.exists():
            print(f"  ⚠️ QC directory not found: {qc_dir}")
            return
            
        qc_dest = self.package_dir / 'qc_analysis'
        
        # Copy QC files with organization
        qc_files = []
        for file_path in qc_source.glob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in ['.png', '.jpg', '.pdf', '.svg']:
                    dest_file = qc_dest / 'plots' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    qc_files.append(f"plots/{file_path.name}")
                else:
                    dest_file = qc_dest / 'data' / file_path.name  
                    shutil.copy2(file_path, dest_file)
                    qc_files.append(f"data/{file_path.name}")
                    
        self.artifacts['qc_analysis'] = {
            'description': description,
            'files': qc_files,
            'source_directory': str(qc_source),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✓ Added {len(qc_files)} QC files")
        
    def add_statistical_analysis(self, stats_dir: str, description: str = "Statistical Analysis"):
        """
        Add statistical analysis results to the package.
        
        Args:
            stats_dir: Directory containing statistical analysis outputs
            description: Description of the statistical analysis
        """
        print(f"📈 Adding statistical analysis from: {stats_dir}")
        
        stats_source = Path(stats_dir)
        if not stats_source.exists():
            print(f"  ⚠️ Statistical analysis directory not found: {stats_dir}")
            return
            
        stats_dest = self.package_dir / 'statistical_analysis'
        
        # Copy statistical files with organization
        stats_files = []
        for file_path in stats_source.glob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in ['.png', '.jpg', '.pdf', '.svg']:
                    dest_file = stats_dest / 'plots' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    stats_files.append(f"plots/{file_path.name}")
                else:
                    dest_file = stats_dest / 'results' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    stats_files.append(f"results/{file_path.name}")
                    
        self.artifacts['statistical_analysis'] = {
            'description': description,
            'files': stats_files,
            'source_directory': str(stats_source),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✓ Added {len(stats_files)} statistical analysis files")
        
    def add_ml_models(self, ml_dir: str, description: str = "Machine Learning Models"):
        """
        Add ML model results to the package.
        
        Args:
            ml_dir: Directory containing ML model outputs
            description: Description of the ML analysis
        """
        print(f"🤖 Adding ML models from: {ml_dir}")
        
        ml_source = Path(ml_dir)
        if not ml_source.exists():
            print(f"  ⚠️ ML directory not found: {ml_dir}")
            return
            
        ml_dest = self.package_dir / 'ml_models'
        
        # Copy ML files with organization
        ml_files = []
        for file_path in ml_source.glob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in ['.pkl', '.joblib', '.model']:
                    dest_file = ml_dest / 'trained_models' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    ml_files.append(f"trained_models/{file_path.name}")
                elif file_path.suffix.lower() in ['.png', '.jpg', '.pdf', '.svg']:
                    dest_file = ml_dest / 'evaluation' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    ml_files.append(f"evaluation/{file_path.name}")
                elif 'importance' in file_path.name.lower():
                    dest_file = ml_dest / 'feature_importance' / file_path.name
                    shutil.copy2(file_path, dest_file)
                    ml_files.append(f"feature_importance/{file_path.name}")
                else:
                    dest_file = ml_dest / file_path.name
                    shutil.copy2(file_path, dest_file)
                    ml_files.append(file_path.name)
                    
        self.artifacts['ml_models'] = {
            'description': description,
            'files': ml_files,
            'source_directory': str(ml_source),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✓ Added {len(ml_files)} ML model files")
        
    def add_data_files(self, data_sources: Dict[str, str], description: str = "Data Files"):
        """
        Add data files to the package.
        
        Args:
            data_sources: Dictionary mapping data types to file paths
            description: Description of the data files
        """
        print(f"📁 Adding data files...")
        
        data_dest = self.package_dir / 'data_files'
        data_files = []
        
        for data_type, file_path in data_sources.items():
            source_path = Path(file_path)
            if source_path.exists():
                if data_type in ['raw', 'processed', 'metadata']:
                    dest_file = data_dest / data_type / source_path.name
                else:
                    dest_file = data_dest / source_path.name
                    
                shutil.copy2(source_path, dest_file)
                data_files.append(f"{data_type}/{source_path.name}" if data_type in ['raw', 'processed', 'metadata'] else source_path.name)
                print(f"  ✓ Added {data_type}: {source_path.name}")
            else:
                print(f"  ⚠️ Data file not found: {file_path}")
                
        self.artifacts['data_files'] = {
            'description': description,
            'files': data_files,
            'sources': data_sources,
            'timestamp': datetime.now().isoformat()
        }
        
    def add_custom_metadata(self, metadata: Dict[str, Any]):
        """
        Add custom metadata to the package.
        
        Args:
            metadata: Dictionary containing custom metadata
        """
        print(f"📋 Adding custom metadata...")
        
        self.artifacts['metadata'].update(metadata)
        
        # Save metadata as JSON
        metadata_file = self.package_dir / 'data_files' / 'metadata' / 'custom_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"  ✓ Added custom metadata with {len(metadata)} entries")
        
    def generate_summary_report(self, analysis_summary: Optional[Dict[str, Any]] = None):
        """
        Generate a comprehensive summary report.
        
        Args:
            analysis_summary: Optional dictionary with analysis results summary
        """
        print(f"📄 Generating summary report...")
        
        summary = {
            'project_name': self.project_name,
            'package_name': self.package_name,
            'generated_at': datetime.now().isoformat(),
            'analysis_summary': analysis_summary or {},
            'artifacts_included': {
                key: len(value.get('files', [])) if isinstance(value, dict) else 0
                for key, value in self.artifacts.items()
            },
            'package_structure': self._get_package_structure()
        }
        
        # Save summary as JSON
        summary_json = self.package_dir / 'reports' / 'json' / 'analysis_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Generate HTML report
        html_report = self._generate_html_report(summary)
        html_file = self.package_dir / 'reports' / 'html' / 'analysis_report.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        # Generate manifest file
        self._generate_manifest()
        
        # Generate README
        self._generate_readme()
        
        self.artifacts['summary'] = summary
        
        print(f"  ✓ Generated summary report")
        
    def _get_package_structure(self) -> Dict[str, List[str]]:
        """Get the structure of the package directory."""
        structure = {}
        
        for root, dirs, files in os.walk(self.package_dir):
            rel_root = os.path.relpath(root, self.package_dir)
            if rel_root == '.':
                rel_root = 'root'
            structure[rel_root] = files
            
        return structure
        
    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate an HTML summary report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Biomarker Discovery Analysis Report - {self.project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .artifact-section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .summary-stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; min-width: 150px; }}
        .file-list {{ max-height: 200px; overflow-y: auto; background: white; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
        ul {{ margin: 0; padding-left: 20px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Biomarker Discovery Analysis Report</h1>
        
        <div class="metadata">
            <h2>📋 Project Information</h2>
            <p><strong>Project Name:</strong> {summary['project_name']}</p>
            <p><strong>Package Name:</strong> {summary['package_name']}</p>
            <p><strong>Generated:</strong> {summary['generated_at']}</p>
        </div>
        
        <h2>📊 Analysis Summary</h2>
        <div class="summary-stats">
"""
        
        # Add analysis summary stats if available
        if summary.get('analysis_summary'):
            for key, value in summary['analysis_summary'].items():
                html += f'<div class="stat-box"><strong>{value}</strong><br>{key.replace("_", " ").title()}</div>'
                
        html += "</div>"
        
        # Add artifacts summary
        html += """
        <h2>📦 Included Artifacts</h2>
"""
        
        for artifact_type, count in summary['artifacts_included'].items():
            if count > 0:
                artifact_info = self.artifacts.get(artifact_type, {})
                description = artifact_info.get('description', artifact_type.replace('_', ' ').title())
                
                html += f"""
        <div class="artifact-section">
            <h3>{description}</h3>
            <p><strong>Files included:</strong> {count}</p>
"""
                
                if 'files' in artifact_info:
                    html += '<div class="file-list"><ul>'
                    for file_name in artifact_info['files'][:10]:  # Show first 10 files
                        html += f'<li>{file_name}</li>'
                    if len(artifact_info['files']) > 10:
                        html += f'<li>... and {len(artifact_info["files"]) - 10} more files</li>'
                    html += '</ul></div>'
                    
                if 'timestamp' in artifact_info:
                    html += f'<p class="timestamp">Added: {artifact_info["timestamp"]}</p>'
                    
                html += '</div>'
                
        html += """
        <h2>📁 Package Structure</h2>
        <div class="file-list">
            <ul>
"""
        
        for folder, files in summary['package_structure'].items():
            if files:  # Only show folders with files
                html += f'<li><strong>{folder}/</strong> ({len(files)} files)</li>'
                
        html += """
            </ul>
        </div>
        
        <h2>📖 Usage Instructions</h2>
        <div class="artifact-section">
            <p>This package contains all artifacts from your biomarker discovery analysis:</p>
            <ul>
                <li><strong>qc_analysis/</strong> - Quality control plots and data</li>
                <li><strong>statistical_analysis/</strong> - Statistical test results and volcano plots</li>
                <li><strong>ml_models/</strong> - Trained machine learning models and evaluation</li>
                <li><strong>data_files/</strong> - Processed data matrices and metadata</li>
                <li><strong>reports/</strong> - Summary reports and documentation</li>
            </ul>
            <p>See the README.md file for detailed information about each artifact.</p>
        </div>
        
        <footer style="margin-top: 40px; text-align: center; color: #7f8c8d; border-top: 1px solid #ddd; padding-top: 20px;">
            <p>Generated by OmniBio Biomarker Discovery Pipeline</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
        
    def _generate_manifest(self):
        """Generate a manifest file listing all package contents."""
        manifest_file = self.package_dir / 'MANIFEST.txt'
        
        with open(manifest_file, 'w') as f:
            f.write(f"BIOMARKER DISCOVERY PACKAGE MANIFEST\n")
            f.write(f"=====================================\n\n")
            f.write(f"Project: {self.project_name}\n")
            f.write(f"Package: {self.package_name}\n") 
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("CONTENTS:\n")
            f.write("---------\n")
            
            for root, dirs, files in os.walk(self.package_dir):
                level = root.replace(str(self.package_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                rel_root = os.path.relpath(root, self.package_dir)
                if rel_root != '.':
                    f.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    if file != 'MANIFEST.txt':  # Don't include the manifest in itself
                        f.write(f"{subindent}{file}\n")
                        
    def _generate_readme(self):
        """Generate a comprehensive README file."""
        readme_file = self.package_dir / 'README.md'
        
        readme_content = f"""# Biomarker Discovery Analysis Package

## Project Information
- **Project Name:** {self.project_name}
- **Package Name:** {self.package_name}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This package contains comprehensive results from a biomarker discovery analysis using the OmniBio pipeline. The analysis includes quality control, statistical testing, machine learning model training, and biomarker ranking.

## Package Structure

### 📊 Quality Control Analysis (`qc_analysis/`)
- **plots/**: TIC/BPC chromatograms, PCA plots, batch effect analysis
- **data/**: Processed QC data and metrics

### 📈 Statistical Analysis (`statistical_analysis/`)
- **plots/**: Volcano plots, statistical visualizations
- **results/**: T-test results, significant features, statistical summaries

### 🤖 Machine Learning Models (`ml_models/`)
- **trained_models/**: Serialized ML models (.pkl files)
- **evaluation/**: ROC curves, performance metrics, validation results
- **feature_importance/**: Feature ranking and importance scores

### 📁 Data Files (`data_files/`)
- **raw/**: Original input data files
- **processed/**: Cleaned and normalized feature matrices
- **metadata/**: Sample information and experimental metadata

### 📄 Reports (`reports/`)
- **html/**: Interactive HTML analysis report
- **json/**: Machine-readable analysis summary

### 📖 Documentation (`documentation/`)
- Additional documentation and methodology notes

## Key Files

### Analysis Report
- `reports/html/analysis_report.html` - Comprehensive HTML report with visualizations
- `reports/json/analysis_summary.json` - Machine-readable analysis summary

### Data Files
- `data_files/processed/` - Final processed feature matrices ready for analysis
- `statistical_analysis/results/` - Complete statistical test results

### Models
- `ml_models/trained_models/` - Trained biomarker classification models
- `ml_models/evaluation/` - Model performance evaluation

## Usage Instructions

### Viewing Results
1. Open `reports/html/analysis_report.html` in a web browser for the main report
2. Review `MANIFEST.txt` for a complete file listing
3. Check `statistical_analysis/results/` for detailed biomarker rankings

### Using Trained Models
The trained models in `ml_models/trained_models/` can be loaded using:

```python
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Data Format
- Feature matrices are saved as CSV files with samples as rows
- Statistical results include p-values, fold changes, and significance flags
- All plots are provided in high-resolution PNG format

## Analysis Pipeline

This analysis was performed using the OmniBio biomarker discovery pipeline, which includes:

1. **Data Ingestion**: Support for mwTab and mzML formats
2. **Quality Control**: TIC/BPC analysis, PCA for batch effects
3. **Statistical Analysis**: T-tests, multiple testing correction, volcano plots
4. **Machine Learning**: Logistic regression with cross-validation
5. **Biomarker Ranking**: Feature importance and statistical significance

## Support

For questions about this analysis or the OmniBio platform, please refer to the documentation or contact support.

---
*Generated by OmniBio Biomarker Discovery Pipeline*
"""
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
            
    def create_zip_package(self, include_data: bool = True) -> str:
        """
        Create a ZIP file of the complete package.
        
        Args:
            include_data: Whether to include large data files
            
        Returns:
            Path to the created ZIP file
        """
        print(f"📦 Creating ZIP package...")
        
        zip_path = self.output_dir / f"{self.package_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Optionally skip large data files
                    if not include_data and file_path.endswith(('.csv', '.pkl')) and os.path.getsize(file_path) > 10*1024*1024:  # 10MB
                        continue
                        
                    arc_name = os.path.relpath(file_path, self.package_dir)
                    zipf.write(file_path, arc_name)
                    
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  ✓ Created ZIP package: {zip_path} ({zip_size_mb:.1f} MB)")
        
        return str(zip_path)
        
    def create_tar_package(self, compression: str = 'gz', include_data: bool = True) -> str:
        """
        Create a TAR file of the complete package.
        
        Args:
            compression: Compression type ('', 'gz', 'bz2', 'xz')
            include_data: Whether to include large data files
            
        Returns:
            Path to the created TAR file
        """
        print(f"📦 Creating TAR package with {compression} compression...")
        
        if compression:
            tar_path = self.output_dir / f"{self.package_name}.tar.{compression}"
            mode = f"w:{compression}"
        else:
            tar_path = self.output_dir / f"{self.package_name}.tar"
            mode = "w"
            
        with tarfile.open(tar_path, mode) as tarf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Optionally skip large data files  
                    if not include_data and file_path.endswith(('.csv', '.pkl')) and os.path.getsize(file_path) > 10*1024*1024:  # 10MB
                        continue
                        
                    arc_name = os.path.relpath(file_path, self.package_dir)
                    tarf.add(file_path, arc_name)
                    
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"  ✓ Created TAR package: {tar_path} ({tar_size_mb:.1f} MB)")
        
        return str(tar_path)
        
    def get_package_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the created package."""
        package_info = {
            'project_name': self.project_name,
            'package_name': self.package_name,
            'package_directory': str(self.package_dir),
            'created_at': datetime.now().isoformat(),
            'artifacts': self.artifacts,
            'total_files': sum(len(files) for _, _, files in os.walk(self.package_dir)),
            'package_size_mb': sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(self.package_dir)
                for file in files
            ) / (1024 * 1024)
        }
        
        return package_info


def create_biomarker_package(
    output_dir: str,
    qc_dir: Optional[str] = None,
    stats_dir: Optional[str] = None, 
    ml_dir: Optional[str] = None,
    data_sources: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    analysis_summary: Optional[Dict[str, Any]] = None,
    project_name: str = "biomarker_discovery",
    create_zip: bool = True,
    create_tar: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to create a complete biomarker analysis package.
    
    Args:
        output_dir: Directory to create the package in
        qc_dir: Directory containing QC analysis results
        stats_dir: Directory containing statistical analysis results
        ml_dir: Directory containing ML model results
        data_sources: Dictionary mapping data types to file paths
        metadata: Custom metadata to include
        analysis_summary: Summary of analysis results
        project_name: Name for the project
        create_zip: Whether to create a ZIP archive
        create_tar: Whether to create a TAR archive
        
    Returns:
        Dictionary with package information and file paths
    """
    
    print("=" * 60)
    print("🎯 BIOMARKER ANALYSIS ARTIFACT PACKAGER")
    print("=" * 60)
    
    # Initialize packager
    packager = ArtifactPackager(output_dir, project_name)
    
    # Add artifacts
    if qc_dir:
        packager.add_qc_analysis(qc_dir)
        
    if stats_dir:
        packager.add_statistical_analysis(stats_dir)
        
    if ml_dir:
        packager.add_ml_models(ml_dir)
        
    if data_sources:
        packager.add_data_files(data_sources)
        
    if metadata:
        packager.add_custom_metadata(metadata)
        
    # Generate reports
    packager.generate_summary_report(analysis_summary)
    
    # Create archives
    result = {
        'package_info': packager.get_package_info(),
        'package_directory': str(packager.package_dir),
        'archives': {}
    }
    
    if create_zip:
        zip_path = packager.create_zip_package()
        result['archives']['zip'] = zip_path
        
    if create_tar:
        tar_path = packager.create_tar_package()
        result['archives']['tar'] = tar_path
        
    print("\n" + "=" * 60)
    print("✅ ARTIFACT PACKAGING COMPLETE")
    print("=" * 60)
    print(f"📦 Package directory: {packager.package_dir}")
    print(f"📁 Total files: {result['package_info']['total_files']}")
    print(f"💾 Package size: {result['package_info']['package_size_mb']:.1f} MB")
    
    if result['archives']:
        print("📦 Archives created:")
        for archive_type, path in result['archives'].items():
            print(f"  {archive_type.upper()}: {path}")
            
    return result 