"""
Data Models for OmniBio
Pydantic models for API requests, responses, and internal data structures.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, validator


class FileType(str, Enum):
    """Supported file types."""
    MWTAB = "mwtab"
    MZML_CENTROID = "mzml_centroid"
    MZML_PROFILE = "mzml_profile"
    MZML_CHROMATOGRAM = "mzml_chromatogram"


class AnalysisType(str, Enum):
    """Available analysis types."""
    QC = "qc"
    STATISTICAL = "statistical"
    ML = "ml"
    PCA = "pca"
    FEATURE_EXTRACTION = "feature_extraction"
    ENRICHMENT = "enrichment"


class AnalysisStatus(str, Enum):
    """Analysis status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FileInfo(BaseModel):
    """Information about uploaded files."""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Detected file type")
    size_bytes: int = Field(..., description="File size in bytes")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    status: str = Field(default="uploaded", description="File status")


class AnalysisRequest(BaseModel):
    """Request to start an analysis."""
    file_ids: List[str] = Field(..., description="List of uploaded file IDs")
    analysis_types: List[AnalysisType] = Field(
        default=[AnalysisType.QC, AnalysisType.STATISTICAL, AnalysisType.ML],
        description="Types of analysis to run"
    )
    project_name: str = Field(default="biomarker_analysis", description="Name for the analysis project")
    group_column: Optional[str] = Field(default=None, description="Column name for sample groups")


class FeatureExtractionParams(BaseModel):
    """Parameters for feature extraction."""
    mass_error_ppm: float = Field(default=5.0, description="Mass error tolerance (ppm)")
    intensity_threshold: float = Field(default=1000.0, description="Minimum intensity threshold")
    rt_tolerance: float = Field(default=30.0, description="RT tolerance for alignment (seconds)")
    mz_tolerance_ppm: float = Field(default=5.0, description="m/z tolerance for deduplication (ppm)")
    rt_tolerance_min: float = Field(default=0.1, description="RT tolerance for deduplication (minutes)")
    keep_strategy: str = Field(default="highest_abundance", description="Strategy for keeping features")
    min_frequency: float = Field(default=0.6, description="Minimum frequency threshold")
    score_method: str = Field(default="total_abundance", description="Scoring method for filtering")
    min_score_percentile: float = Field(default=50.0, description="Minimum score percentile")
    imputation_method: str = Field(default="median_per_cohort", description="Missing value imputation method")
    scaling_method: str = Field(default="pareto", description="Scaling method")
    log_transform: bool = Field(default=False, description="Apply log transformation")


class StatisticalParams(BaseModel):
    """Parameters for statistical analysis."""
    test_type: str = Field(default="ttest", description="Statistical test type")
    p_threshold: float = Field(default=0.05, description="P-value threshold")
    fc_threshold: float = Field(default=1.0, description="Fold-change threshold")
    fdr_method: str = Field(default="fdr_bh", description="FDR correction method")


class MLParams(BaseModel):
    """Parameters for machine learning."""
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    random_state: int = Field(default=42, description="Random seed")
    models: List[str] = Field(default=["logistic_regression"], description="Models to train")


class EnrichmentParams(BaseModel):
    """Parameters for pathway enrichment analysis."""
    p_value_threshold: float = Field(default=0.05, description="P-value threshold")
    fdr_method: str = Field(default="fdr_bh", description="FDR correction method")
    use_kegg: bool = Field(default=True, description="Include KEGG pathways")
    use_go: bool = Field(default=True, description="Include GO terms")
    organism: str = Field(default="hsa", description="Organism code")
    min_genes_per_pathway: int = Field(default=3, description="Minimum genes per pathway")


class AnalysisResult(BaseModel):
    """Results from a completed analysis."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")
    message: str = Field(..., description="Status message")
    started_at: datetime = Field(..., description="Analysis start time")
    completed_at: Optional[datetime] = Field(default=None, description="Analysis completion time")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Analysis results")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DataInfo(BaseModel):
    """Information about loaded data."""
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    file_ids: List[str] = Field(..., description="Source file IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from files")


class QCResult(BaseModel):
    """Quality control analysis results."""
    qc_plot: Optional[str] = Field(default=None, description="Path to QC plot")
    completed: bool = Field(..., description="Whether analysis completed successfully")


class StatisticalResult(BaseModel):
    """Statistical analysis results."""
    summary: Dict[str, Any] = Field(..., description="Statistical summary")
    files: Dict[str, str] = Field(..., description="Generated file paths")
    pathway_results: Optional[Dict[str, Any]] = Field(default=None, description="Pathway analysis results")
    n_features_tested: int = Field(..., description="Number of features tested")
    n_significant: int = Field(..., description="Number of significant features")
    top_significant_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top significant features")
    completed: bool = Field(..., description="Whether analysis completed successfully")


class MLResult(BaseModel):
    """Machine learning results."""
    models: Dict[str, Dict[str, Any]] = Field(..., description="Results for each trained model")
    completed: bool = Field(..., description="Whether analysis completed successfully")


class PackageRequest(BaseModel):
    """Request to create an artifact package."""
    analysis_id: str = Field(..., description="Analysis ID to package")
    include_data: bool = Field(default=True, description="Whether to include large data files")
    create_zip: bool = Field(default=True, description="Whether to create ZIP archive")
    create_tar: bool = Field(default=False, description="Whether to create TAR archive")


class ReportRequest(BaseModel):
    """Request to generate a report."""
    analysis_id: str = Field(..., description="Analysis ID to generate report from")
    project_name: str = Field(default="Biomarker Analysis", description="Project name for the report")
    author: str = Field(default="Research Team", description="Report author")
    institution: str = Field(default="", description="Institution name")
    theme: str = Field(default="professional", description="Report theme")
    technical_level: str = Field(default="standard", description="Technical detail level")
    include_methodology: bool = Field(default=True, description="Include methodology section")
    include_statistical_analysis: bool = Field(default=True, description="Include statistical analysis")
    include_pathway_analysis: bool = Field(default=True, description="Include pathway enrichment")
    include_quality_control: bool = Field(default=True, description="Include QC assessment")
    generate_html: bool = Field(default=True, description="Generate HTML report")
    generate_pdf: bool = Field(default=False, description="Generate PDF report")


class JobRecord(BaseModel):
    """Database record for analysis jobs."""
    id: Optional[int] = Field(default=None, description="Database ID")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    project_name: str = Field(..., description="Project name")
    status: AnalysisStatus = Field(..., description="Current status")
    analysis_types: List[str] = Field(..., description="Types of analysis requested")
    file_ids: List[str] = Field(..., description="Input file IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    progress: float = Field(default=0.0, description="Progress percentage")
    message: str = Field(default="Queued", description="Status message")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    results_path: Optional[str] = Field(default=None, description="Path to results directory")
    
    class Config:
        from_attributes = True


class HealthStatus(BaseModel):
    """API health status."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    database_connected: bool = Field(..., description="Database connection status")
    storage_accessible: bool = Field(..., description="Storage accessibility status") 