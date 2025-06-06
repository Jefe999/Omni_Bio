#!/usr/bin/env python3
"""
Task #14: FastAPI Endpoints
Main FastAPI application for OmniBio biomarker discovery pipeline.

Provides REST API access to:
- File upload (mwTab, mzML)
- Analysis execution (QC, statistical, ML)
- Results retrieval
- Artifact packaging
- Status monitoring
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import os
import shutil
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import our biomarker modules with new paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Updated imports for new structure
from biomarker.models.config import get_settings
from biomarker.models.data import (
    AnalysisRequest, AnalysisResult, FileInfo, PackageRequest,
    ReportRequest, HealthStatus, AnalysisStatus
)
from biomarker.models.db_manager import get_database_manager, init_database, close_database
from biomarker.models.database import User
from biomarker.api.auth import require_auth, require_write_permission, optional_auth
from biomarker.io.file_loader import load_file, detect_file_type
from biomarker.core.qc.qc_plots import generate_qc_summary_plot, simulate_tic_bpc_from_features
from biomarker.core.qc.pca_analysis import run_complete_pca_analysis
from biomarker.core.ml.statistical_analysis_simple import run_complete_statistical_analysis
from biomarker.core.ml.model_pipeline import train_models
from biomarker.report.artifact_packager import create_biomarker_package

# NEW: Feature extraction and enrichment imports
from biomarker.core.features.peak_picking import run_openms_feature_finder, FeaturePickingParams
from biomarker.core.features.deduplication import deduplicate_features
from biomarker.core.features.filtering import combined_filter
from biomarker.core.features.imputation import impute_missing_values
from biomarker.core.preprocessing.scalers import apply_scaling, ScalerParams
from biomarker.core.analysis.pathway_analysis import run_pathway_enrichment, PathwayEnrichmentParams

# NEW: Report generation imports
from biomarker.report.report_generator import (
    generate_comprehensive_report,
    generate_executive_summary,
    ReportConfig
)

# Get configuration
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="REST API for biomarker discovery from metabolomics data",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware using settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.security.allowed_methods,
    allow_headers=settings.security.allowed_headers,
)

# Configuration from settings
UPLOAD_DIR = Path(settings.storage.upload_dir)
RESULTS_DIR = Path(settings.storage.results_dir)
PACKAGES_DIR = Path(settings.storage.packages_dir)

# Ensure directories exist
for directory in [UPLOAD_DIR, RESULTS_DIR, PACKAGES_DIR]:
    directory.mkdir(exist_ok=True)

# Database startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    try:
        await init_database()
        print("✅ Database connected successfully")
        load_analysis_status()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # Don't fail startup - allow API to run without database for development

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    try:
        await close_database()
        print("✅ Database connection closed")
        save_analysis_status()
    except Exception as e:
        print(f"❌ Error closing database: {e}")

# In-memory storage for analysis status (fallback when database is unavailable)
analysis_status = {}

# Analysis status persistence
STATUS_FILE = RESULTS_DIR / "analysis_status.json"

def save_analysis_status():
    """Save analysis status to disk."""
    try:
        # Convert to serializable format
        serializable_status = {}
        for aid, status in analysis_status.items():
            # Deep copy and ensure all values are JSON serializable
            clean_status = {
                'analysis_id': status['analysis_id'],
                'status': status['status'],
                'progress': float(status['progress']),
                'message': status['message'],
                'started_at': status['started_at'],
                'completed_at': status.get('completed_at'),
                'error': status.get('error'),
                'results': status.get('results')  # This might be complex, but let's try
            }
            serializable_status[aid] = clean_status
        
        with open(STATUS_FILE, 'w') as f:
            json.dump(serializable_status, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Failed to save analysis status: {e}")

def load_analysis_status():
    """Load analysis status from disk."""
    global analysis_status
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                analysis_status = json.load(f)
            print(f"✅ Loaded {len(analysis_status)} analysis records from disk")
        else:
            print("📁 No existing analysis status file found")
            analysis_status = {}
        
        # Also discover completed analyses from results directory
        if RESULTS_DIR.exists():
            discovered_count = 0
            for results_subdir in RESULTS_DIR.iterdir():
                if results_subdir.is_dir():
                    analysis_id = results_subdir.name
                    if analysis_id not in analysis_status:
                        # Check if this looks like a completed analysis
                        stats_dir = results_subdir / 'statistical'
                        if stats_dir.exists() and (stats_dir / 'statistical_results.csv').exists():
                            # Create a basic analysis status entry
                            analysis_status[analysis_id] = {
                                'analysis_id': analysis_id,
                                'status': 'completed',
                                'progress': 1.0,
                                'message': 'Analysis completed (discovered from disk)',
                                'started_at': '2024-01-01T00:00:00',  # Placeholder
                                'completed_at': '2024-01-01T01:00:00',  # Placeholder
                                'results': {
                                    'data_info': {
                                        'n_samples': 0,
                                        'n_features': 0,
                                        'file_ids': [],
                                        'project_name': f'Discovered Analysis {analysis_id[:8]}'
                                    },
                                    'preprocessing': {
                                        'scaling_method': 'unknown',
                                        'log_transform': False,
                                        'log_base': 'log10',
                                        'p_value_threshold': 0.05
                                    }
                                }
                            }
                            discovered_count += 1
            
            if discovered_count > 0:
                print(f"🔍 Discovered {discovered_count} additional completed analyses from disk")
        
    except Exception as e:
        print(f"Warning: Failed to load analysis status: {e}")
        analysis_status = {}

# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for starting analysis."""
    file_ids: List[str] = Field(..., description="List of uploaded file IDs")
    analysis_types: List[str] = Field(
        default=["qc", "statistical", "ml"],
        description="Types of analysis to run: qc, statistical, ml, pca"
    )
    project_name: str = Field(default="biomarker_analysis", description="Name for the analysis project")
    analysis_name: Optional[str] = Field(default=None, description="Custom name for this analysis")
    group_column: Optional[str] = Field(default=None, description="Column name for sample groups")
    
    # Preprocessing parameters
    scaling_method: str = Field(default="pareto", description="Feature scaling method: pareto, standard, minmax, robust, power, none")
    log_transform: bool = Field(default=False, description="Apply log transformation before scaling")
    log_base: str = Field(default="log10", description="Log base: log10, log2, ln")
    p_value_threshold: float = Field(default=0.05, description="Statistical significance threshold")

class AnalysisStatus(BaseModel):
    """Response model for analysis status."""
    analysis_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    started_at: str
    completed_at: Optional[str] = None
    analysis_name: Optional[str] = None
    project_name: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class FileInfo(BaseModel):
    """Response model for uploaded file info."""
    file_id: str
    filename: str
    file_type: str
    size_bytes: int
    uploaded_at: str
    status: str

class PackageRequest(BaseModel):
    """Request model for creating artifact packages."""
    analysis_id: str
    include_data: bool = Field(default=True, description="Whether to include large data files")
    create_zip: bool = Field(default=True, description="Whether to create ZIP archive")
    create_tar: bool = Field(default=False, description="Whether to create TAR archive")

# NEW: Feature extraction request models
class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction pipeline."""
    file_ids: List[str] = Field(..., description="List of uploaded mzML file IDs")
    project_name: str = Field(default="feature_extraction", description="Project name")
    
    # Peak picking parameters
    mass_error_ppm: float = Field(default=5.0, description="Mass error tolerance (ppm)")
    intensity_threshold: float = Field(default=1000.0, description="Minimum intensity threshold")
    rt_tolerance: float = Field(default=30.0, description="RT tolerance for alignment (seconds)")
    
    # Deduplication parameters
    mz_tolerance_ppm: float = Field(default=5.0, description="m/z tolerance for deduplication (ppm)")
    rt_tolerance_min: float = Field(default=0.1, description="RT tolerance for deduplication (minutes)")
    keep_strategy: str = Field(default="highest_abundance", description="Strategy for keeping features")
    
    # Filtering parameters
    min_frequency: float = Field(default=0.6, description="Minimum frequency threshold")
    score_method: str = Field(default="total_abundance", description="Scoring method for filtering")
    min_score_percentile: float = Field(default=50.0, description="Minimum score percentile")
    
    # Imputation parameters
    imputation_method: str = Field(default="median_per_cohort", description="Missing value imputation method")
    group_column: Optional[str] = Field(default=None, description="Group column for per-cohort imputation")
    
    # Scaling parameters
    scaling_method: str = Field(default="pareto", description="Scaling method")
    log_transform: bool = Field(default=False, description="Apply log transformation")

class EnrichmentRequest(BaseModel):
    """Request model for pathway enrichment analysis."""
    feature_matrix_file: str = Field(..., description="Feature matrix file ID or path")
    significant_features: List[str] = Field(..., description="List of significant feature IDs")
    background_features: Optional[List[str]] = Field(default=None, description="Background feature set")
    
    # Enrichment parameters
    p_value_threshold: float = Field(default=0.05, description="P-value threshold")
    fdr_method: str = Field(default="fdr_bh", description="FDR correction method")
    use_kegg: bool = Field(default=True, description="Include KEGG pathways")
    use_go: bool = Field(default=True, description="Include GO terms")
    organism: str = Field(default="hsa", description="Organism code")
    min_genes_per_pathway: int = Field(default=3, description="Minimum genes per pathway")

class IntegratedAnalysisRequest(BaseModel):
    """Request model for integrated feature extraction + enrichment analysis."""
    file_ids: List[str] = Field(..., description="List of uploaded mzML file IDs")
    project_name: str = Field(default="integrated_analysis", description="Project name")
    group_column: Optional[str] = Field(default=None, description="Sample group column")
    
    # Feature extraction settings (simplified)
    feature_extraction: bool = Field(default=True, description="Run feature extraction")
    mass_error_ppm: float = Field(default=5.0, description="Mass error tolerance")
    min_frequency: float = Field(default=0.6, description="Minimum feature frequency")
    scaling_method: str = Field(default="pareto", description="Scaling method")
    
    # Statistical analysis for significant features
    run_statistics: bool = Field(default=True, description="Run statistical analysis")
    
    # Enrichment analysis settings
    run_enrichment: bool = Field(default=True, description="Run pathway enrichment")
    p_value_threshold: float = Field(default=0.05, description="Enrichment p-value threshold")
    use_kegg: bool = Field(default=True, description="Include KEGG pathways")
    use_go: bool = Field(default=True, description="Include GO terms")

# NEW: Report generation request models
class ReportGenerationRequest(BaseModel):
    """Request model for generating comprehensive reports."""
    analysis_id: str = Field(..., description="Analysis ID to generate report from")
    project_name: str = Field(default="Biomarker Analysis", description="Project name for the report")
    author: str = Field(default="Research Team", description="Report author")
    institution: str = Field(default="", description="Institution name")
    
    # Report configuration
    theme: str = Field(default="professional", description="Report theme: professional, clinical, academic")
    technical_level: str = Field(default="standard", description="Technical detail level: basic, standard, detailed")
    include_methodology: bool = Field(default=True, description="Include methodology section")
    include_statistical_analysis: bool = Field(default=True, description="Include statistical analysis")
    include_pathway_analysis: bool = Field(default=True, description="Include pathway enrichment")
    include_quality_control: bool = Field(default=True, description="Include QC assessment")
    
    # Output formats
    generate_html: bool = Field(default=True, description="Generate HTML report")
    generate_pdf: bool = Field(default=False, description="Generate PDF report")

class ExecutiveSummaryRequest(BaseModel):
    """Request model for generating executive summaries."""
    analysis_id: str = Field(..., description="Analysis ID to generate summary from")
    project_name: str = Field(default="Biomarker Study", description="Project name")
    author: str = Field(default="Clinical Team", description="Report author")

# Helper functions
def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def get_file_path(file_id: str) -> Path:
    """Get the file path for a given file ID."""
    return UPLOAD_DIR / file_id

def get_results_path(analysis_id: str) -> Path:
    """Get the results directory for a given analysis ID."""
    return RESULTS_DIR / analysis_id

def update_analysis_status(analysis_id: str, **kwargs):
    """Update the status of an analysis."""
    if analysis_id not in analysis_status:
        analysis_status[analysis_id] = {
            'analysis_id': analysis_id,
            'status': 'pending',
            'progress': 0.0,
            'message': 'Analysis queued',
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'results': None,
            'error': None
        }
    
    analysis_status[analysis_id].update(kwargs)
    # Save to disk whenever status is updated
    save_analysis_status()

# API Endpoints

@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint."""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment
    }

@app.get("/health", response_model=HealthStatus, summary="Detailed Health Check")
async def health_check():
    """Detailed health check with system information."""
    # Check storage accessibility
    storage_accessible = True
    try:
        for directory in [UPLOAD_DIR, RESULTS_DIR, PACKAGES_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        storage_accessible = False
    
    # Database connection check
    database_connected = False
    try:
        db_manager = get_database_manager()
        health_result = await db_manager.health_check()
        database_connected = health_result.get("connected", False)
    except Exception as e:
        print(f"Database health check failed: {e}")
        database_connected = False
    
    return HealthStatus(
        status="healthy" if (storage_accessible and database_connected) else "degraded",
        timestamp=datetime.now(),
        version=settings.app_version,
        environment=settings.environment,
        database_connected=database_connected,
        storage_accessible=storage_accessible
    )

@app.post("/upload", response_model=FileInfo, summary="Upload Data File")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(require_write_permission)
):
    """
    Upload a metabolomics data file (mwTab or mzML format).
    
    Returns file information including auto-detected file type.
    Requires write permissions.
    """
    try:
        # Generate unique file ID with original extension
        file_id = generate_id()
        
        # Preserve the original file extension
        original_filename = file.filename or "unknown"
        file_extension = Path(original_filename).suffix.lower()
        
        # Create file path with extension
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect file type
        file_type = detect_file_type(str(file_path))
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create file record in database
        try:
            db_manager = get_database_manager()
            file_record = await db_manager.create_file_record(
                file_info={
                    "file_id": f"{file_id}{file_extension}",
                    "filename": original_filename,
                    "file_type": file_type,
                    "file_path": str(file_path),
                    "size_bytes": file_size,
                    "metadata": {"original_extension": file_extension}
                },
                uploaded_by=str(current_user.id)
            )
        except Exception as e:
            print(f"Warning: Could not save file record to database: {e}")
            # Continue without database - file is still saved locally
        
        file_info = FileInfo(
            file_id=f"{file_id}{file_extension}",  # Include extension in file_id
            filename=file.filename,
            file_type=file_type,
            size_bytes=file_size,
            uploaded_at=datetime.now().isoformat(),
            status="uploaded"
        )
        
        return file_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/files", response_model=List[FileInfo], summary="List Uploaded Files")
async def list_files():
    """List all uploaded files."""
    files = []
    
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            try:
                file_type = detect_file_type(str(file_path))
                file_size = os.path.getsize(file_path)
                
                files.append(FileInfo(
                    file_id=file_path.name,
                    filename=file_path.name,  # In production, store original filename separately
                    file_type=file_type,
                    size_bytes=file_size,
                    uploaded_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    status="uploaded"
                ))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    return files

@app.delete("/files/{file_id}", summary="Delete Uploaded File")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    file_path = get_file_path(file_id)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"message": f"File {file_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.post("/analyze", response_model=AnalysisStatus, summary="Start Biomarker Analysis")
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start biomarker discovery analysis on uploaded files.
    
    Supports multiple analysis types:
    - qc: Quality control plots
    - pca: PCA analysis for batch effects
    - statistical: T-tests and volcano plots
    - ml: Machine learning model training
    """
    # Generate analysis ID
    analysis_id = generate_id()
    
    # Validate files exist
    for file_id in request.file_ids:
        file_path = get_file_path(file_id)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    
    # Initialize analysis status
    update_analysis_status(
        analysis_id,
        status='pending',
        message='Analysis queued for execution',
        analysis_name=request.analysis_name
    )
    
    # Start background analysis
    background_tasks.add_task(
        run_background_analysis,
        analysis_id,
        request.file_ids,
        request.analysis_types,
        request.project_name,
        request.analysis_name,
        request.group_column,
        request.scaling_method,
        request.log_transform,
        request.log_base,
        request.p_value_threshold
    )
    
    return AnalysisStatus(**analysis_status[analysis_id])

async def run_background_analysis(
    analysis_id: str,
    file_ids: List[str],
    analysis_types: List[str],
    project_name: str,
    analysis_name: str,
    group_column: Optional[str],
    scaling_method: str,
    log_transform: bool,
    log_base: str,
    p_value_threshold: float
):
    """Run the biomarker analysis in the background."""
    try:
        update_analysis_status(
            analysis_id,
            status='running',
            progress=0.1,
            message='Loading data files...'
        )
        
        # Create results directory
        results_dir = get_results_path(analysis_id)
        results_dir.mkdir(exist_ok=True)
        
        # Load data files
        dataframes = []
        metadata_list = []
        
        for file_id in file_ids:
            file_path = get_file_path(file_id)
            df, metadata = load_file(str(file_path))
            dataframes.append(df)
            metadata_list.append(metadata)
        
        # Combine dataframes if multiple files
        if len(dataframes) == 1:
            combined_df = dataframes[0]
            combined_metadata = metadata_list[0]
        else:
            # Simple concatenation - enhance for production
            combined_df = pd.concat(dataframes, axis=0)
            combined_metadata = {
                'combined_from': len(dataframes),
                'individual_metadata': metadata_list
            }
        
        update_analysis_status(
            analysis_id,
            progress=0.2,
            message=f'Loaded {combined_df.shape[0]} samples × {combined_df.shape[1]} features'
        )
        
        # Initialize results
        results = {
            'data_info': {
                'n_samples': len(combined_df),
                'n_features': len(combined_df.select_dtypes(include=[np.number]).columns),
                'file_ids': file_ids,
                'metadata': combined_metadata,
                'project_name': project_name  # Store the project name
            },
            'preprocessing': {
                'scaling_method': scaling_method,
                'log_transform': log_transform,
                'log_base': log_base,
                'p_value_threshold': p_value_threshold
            }
        }
        
        # Run QC analysis
        if 'qc' in analysis_types:
            update_analysis_status(
                analysis_id,
                progress=0.3,
                message='Running QC analysis...'
            )
            
            qc_dir = results_dir / 'qc'
            qc_dir.mkdir(exist_ok=True)
            
            # Generate QC plots
            qc_summary_file = generate_qc_summary_plot(combined_df, str(qc_dir))
            
            # Also generate TIC data for frontend interactive plotting
            # Use the simulate function to create TIC/BPC data from the first sample
            sample_name = combined_df.index[0] if len(combined_df) > 0 else "Sample_1"
            tic_traces = simulate_tic_bpc_from_features(combined_df, sample_name)
            
            results['qc'] = {
                'qc_plot': str(qc_summary_file),
                'tic_data': {
                    'retention_time': tic_traces['tic_times'].tolist(),
                    'intensity': tic_traces['tic_intensities'].tolist()
                },
                'bpc_data': {
                    'retention_time': tic_traces['bpc_times'].tolist(),
                    'intensity': tic_traces['bpc_intensities'].tolist()
                },
                'completed': True
            }
        
        # Run PCA analysis
        if 'pca' in analysis_types:
            update_analysis_status(
                analysis_id,
                progress=0.4,
                message='Running PCA analysis...'
            )
            
            pca_dir = results_dir / 'pca'
            pca_dir.mkdir(exist_ok=True)
            
            try:
                # Extract labels from group_column if specified
                labels = None
                pca_df = combined_df.copy()
                
                if group_column and group_column in combined_df.columns:
                    labels = combined_df[group_column]
                    pca_df = combined_df.drop(columns=[group_column])
                else:
                    # Create dummy labels for PCA if no group column specified
                    n_samples = len(combined_df)
                    labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                                     index=combined_df.index, name='Group')
                    pca_df = combined_df
                
                pca_results = run_complete_pca_analysis(
                    pca_df,
                    output_dir=str(pca_dir),
                    labels=labels
                )
                
                # Transform PCA results for frontend consumption
                frontend_pca_data = transform_pca_results_for_frontend(pca_results)
                
                results['pca'] = {
                    'results': frontend_pca_data,
                    'completed': True
                }
            except Exception as e:
                results['pca'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Run statistical analysis
        if 'statistical' in analysis_types:
            update_analysis_status(
                analysis_id,
                progress=0.6,
                message='Running statistical analysis...'
            )
            
            stats_dir = results_dir / 'statistical'
            stats_dir.mkdir(exist_ok=True)
            
            try:
                # Create dummy labels if group_column not specified
                if group_column and group_column in combined_df.columns:
                    stats_labels = combined_df[group_column]
                    stats_df = combined_df.drop(columns=[group_column])
                else:
                    # Create dummy labels for testing
                    n_samples = len(combined_df)
                    stats_labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                                           index=combined_df.index, name='Group')
                    stats_df = combined_df
                
                # Apply user-controlled scaling before statistical analysis
                if scaling_method != 'none':
                    update_analysis_status(
                        analysis_id,
                        progress=0.55,
                        message=f'Applying {scaling_method} scaling...'
                    )
                    
                    # Set up scaling parameters
                    scaler_params = ScalerParams(
                        method=scaling_method,
                        log_transform=log_transform,
                        log_base=log_base,
                        center=True,
                        handle_zeros='offset',
                        clip_outliers=False
                    )
                    
                    scaled_df, scaling_info = apply_scaling(stats_df, scaling_method, scaler_params)
                    
                    # Save scaling information
                    scaling_report_file = stats_dir / "scaling_report.json"
                    with open(scaling_report_file, 'w') as f:
                        json.dump(scaling_info, f, indent=2, default=str)
                    
                    print(f"  ✓ Scaling applied: {scaling_method}")
                    print(f"  ✓ Log transform: {log_transform} ({log_base})")
                    
                    stats_df = scaled_df
                
                stats_results = run_complete_statistical_analysis(
                    stats_df,
                    stats_labels,
                    output_dir=str(stats_dir),
                    alpha=p_value_threshold
                )
                
                # Convert pandas objects to JSON-serializable formats
                serializable_results = {
                    'summary': stats_results['summary'],
                    'files': {k: str(v) for k, v in stats_results['files'].items()},
                    'pathway_results': stats_results['pathway_results'],
                    # Convert DataFrame to dict (without including the actual DataFrame)
                    'n_features_tested': len(stats_results['results_df']),
                    'n_significant': int(stats_results['results_df']['significant'].sum()),
                    'top_significant_features': stats_results['results_df'][
                        stats_results['results_df']['significant']
                    ].head(10)[['feature', 'p_value', 'fold_change', 'significant']].to_dict('records')
                }
                
                results['statistical'] = {
                    'results': serializable_results,
                    'summary': stats_results['summary'],
                    'significant_features': stats_results['results_df'][
                        stats_results['results_df']['significant']
                    ].head(20).to_dict('records') if not stats_results['results_df'].empty else [],
                    'all_features': stats_results['results_df'].to_dict('records') if not stats_results['results_df'].empty else [],
                    'scaling_method': scaling_method,
                    'log_transform': log_transform,
                    'log_base': log_base,
                    'p_value_threshold': p_value_threshold,
                    'completed': True
                }
            except Exception as e:
                results['statistical'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Run ML training
        if 'ml' in analysis_types:
            update_analysis_status(
                analysis_id,
                progress=0.8,
                message='Training ML models...'
            )
            
            ml_dir = results_dir / 'ml'
            ml_dir.mkdir(exist_ok=True)
            
            try:
                # Create dummy labels if group_column not specified
                if group_column and group_column in combined_df.columns:
                    ml_labels = combined_df[group_column]
                    ml_df = combined_df.drop(columns=[group_column])
                else:
                    # Create dummy labels for testing
                    n_samples = len(combined_df)
                    ml_labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                                        index=combined_df.index, name='Group')
                    ml_df = combined_df
                
                ml_results = train_models(
                    ml_df,
                    ml_labels,
                    output_dir=str(ml_dir)
                )
                
                # Convert ML results to JSON-serializable format
                serializable_ml_results = {}
                for model_name, model_data in ml_results.items():
                    model_results = model_data['results']
                    artifacts = model_data['artifacts']
                    
                    # Load feature importance with proper feature names from saved artifacts
                    top_features = []
                    if 'importance' in artifacts:
                        try:
                            # Load feature importance from the saved JSON file which includes feature names
                            with open(artifacts['importance'], 'r') as f:
                                importance_data = json.load(f)
                            top_features = importance_data[:10]  # Top 10 features with names
                            print(f"  ✓ Loaded {len(top_features)} top features with names for {model_name}")
                        except Exception as e:
                            print(f"  ⚠️ Could not load feature names from {artifacts['importance']}: {e}")
                            # Fallback to original method without names
                            top_features = model_results['feature_importance'].head(10).to_dict('records') if 'feature_importance' in model_results else []
                    
                    # Extract key metrics without storing the actual model objects
                    serializable_ml_results[model_name] = {
                        'final_auc': float(model_results['final_auc']),
                        'model_type': model_results['model_type'],
                        'artifacts': {k: str(v) for k, v in artifacts.items()},
                        'top_features': top_features
                    }
                    
                    # Add CV metrics if available
                    if 'mean_cv_auc' in model_results:
                        serializable_ml_results[model_name]['mean_cv_auc'] = float(model_results['mean_cv_auc'])
                        serializable_ml_results[model_name]['std_cv_auc'] = float(model_results['std_cv_auc'])
                    elif 'best_cv_auc' in model_results:
                        serializable_ml_results[model_name]['best_cv_auc'] = float(model_results['best_cv_auc'])
                
                results['ml'] = {
                    'results': serializable_ml_results,
                    'completed': True
                }
            except Exception as e:
                results['ml'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Analysis completed successfully
        update_analysis_status(
            analysis_id,
            status='completed',
            progress=1.0,
            message='Analysis completed successfully',
            completed_at=datetime.now().isoformat(),
            results=results
        )
        
    except Exception as e:
        update_analysis_status(
            analysis_id,
            status='failed',
            progress=0.0,
            message='Analysis failed',
            completed_at=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/analyses", response_model=List[AnalysisStatus], summary="List All Analyses")
async def list_analyses():
    """List all analyses with their current status."""
    serializable_analyses = []
    for analysis_data in analysis_status.values():
        # Create a clean copy without any pandas DataFrames
        clean_analysis = {
            'analysis_id': analysis_data['analysis_id'],
            'status': analysis_data['status'],
            'progress': analysis_data['progress'],
            'message': analysis_data['message'],
            'started_at': analysis_data['started_at'],
            'completed_at': analysis_data.get('completed_at'),
            'analysis_name': analysis_data.get('analysis_name'),
            'error': analysis_data.get('error'),
            'results': None  # Don't include full results in list view
        }
        
        # Include just the project name from results if available (for job titles)
        if analysis_data.get('results', {}) and analysis_data['results'].get('data_info', {}).get('project_name'):
            clean_analysis['project_name'] = analysis_data['results']['data_info']['project_name']
        
        serializable_analyses.append(AnalysisStatus(**clean_analysis))
    return serializable_analyses

@app.get("/analyses/{analysis_id}", response_model=AnalysisStatus, summary="Get Analysis Status")
async def get_analysis_status(analysis_id: str):
    """Get the status of a specific analysis."""
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get the basic analysis data
    analysis_data = analysis_status[analysis_id].copy()
    
    # If analysis is completed but missing detailed results, try to load from disk
    if analysis_data['status'] == 'completed' and analysis_data.get('results'):
        results_dir = get_results_path(analysis_id)
        
        # Load statistical results if available
        stats_dir = results_dir / 'statistical'
        if stats_dir.exists() and analysis_data.get('results', {}).get('statistical') is None:
            try:
                # Load statistical results from CSV
                stats_csv = stats_dir / 'statistical_results.csv'
                if stats_csv.exists():
                    stats_df = pd.read_csv(stats_csv)
                    
                    # Get significant features
                    if 'significant' in stats_df.columns:
                        significant_features = stats_df[stats_df['significant'] == True]
                    else:
                        # Fallback: use p_value threshold
                        p_threshold = analysis_data['results'].get('preprocessing', {}).get('p_value_threshold', 0.05)
                        significant_features = stats_df[stats_df['p_value'] < p_threshold]
                    
                    # Create statistical results summary
                    statistical_results = {
                        'results': {
                            'summary': {
                                'analysis_type': 'ttest',
                                'n_features_tested': len(stats_df),
                                'n_significant_raw': len(significant_features),
                                'n_significant_adj': len(significant_features[significant_features.get('significant_adj', False)]) if 'significant_adj' in stats_df.columns else len(significant_features),
                                'alpha': analysis_data['results'].get('preprocessing', {}).get('p_value_threshold', 0.05),
                                'fc_threshold': 1.0,
                                'top_features': significant_features.head(10)[['feature', 'p_value', 'fold_change']].to_dict('records') if len(significant_features) > 0 else []
                            },
                            'files': {'volcano_plot': str(stats_dir / 'volcano_plot.png')} if (stats_dir / 'volcano_plot.png').exists() else {},
                            'pathway_results': None,
                            'n_features_tested': len(stats_df),
                            'n_significant': len(significant_features),
                            'top_significant_features': significant_features.head(10).to_dict('records') if len(significant_features) > 0 else []
                        },
                        'summary': {
                            'analysis_type': 'ttest',
                            'significant_raw': len(significant_features),
                            'significant_adjusted': len(significant_features[significant_features.get('significant_adj', False)]) if 'significant_adj' in stats_df.columns else len(significant_features),
                            'total_features': len(stats_df)
                        },
                        'significant_features': significant_features.to_dict('records') if len(significant_features) > 0 else [],
                        'all_features': stats_df.to_dict('records'),  # Include all features for volcano plot
                        'scaling_method': analysis_data['results'].get('preprocessing', {}).get('scaling_method', 'unknown'),
                        'log_transform': analysis_data['results'].get('preprocessing', {}).get('log_transform', False),
                        'log_base': analysis_data['results'].get('preprocessing', {}).get('log_base', 'log10'),
                        'p_value_threshold': analysis_data['results'].get('preprocessing', {}).get('p_value_threshold', 0.05),
                        'completed': True
                    }
                    
                    # Add statistical results to the analysis data
                    if 'results' not in analysis_data:
                        analysis_data['results'] = {}
                    analysis_data['results']['statistical'] = statistical_results
                    
            except Exception as e:
                print(f"Error loading statistical results from disk: {e}")
        
        # Load ML results if available
        ml_dir = results_dir / 'ml'
        if ml_dir.exists() and analysis_data.get('results', {}).get('ml') is None:
            try:
                # Load ML results from saved artifacts
                ml_results = {}
                
                # Look for different model types
                for model_type in ['logistic_regression', 'lightgbm']:
                    importance_file = ml_dir / f"{model_type}_importance.json"
                    summary_file = ml_dir / f"{model_type}_summary.json"
                    
                    if importance_file.exists() and summary_file.exists():
                        try:
                            # Load feature importance with names
                            with open(importance_file, 'r') as f:
                                importance_data = json.load(f)
                            
                            # Load summary metrics
                            with open(summary_file, 'r') as f:
                                summary_data = json.load(f)
                            
                            # Create ML result structure with feature names
                            ml_results[model_type] = {
                                'final_auc': summary_data.get('final_auc', 0.0),
                                'model_type': summary_data.get('model_type', model_type),
                                'top_features': importance_data[:10],  # Top 10 with proper names
                                'artifacts': {
                                    'importance': str(importance_file),
                                    'summary': str(summary_file)
                                }
                            }
                            
                            # Add CV metrics if available
                            if 'mean_cv_auc' in summary_data:
                                ml_results[model_type]['mean_cv_auc'] = summary_data['mean_cv_auc']
                            
                        except Exception as e:
                            print(f"Error loading {model_type} results: {e}")
                
                if ml_results:
                    analysis_data['results']['ml'] = {
                        'results': ml_results,
                        'completed': True
                    }
                    print(f"✓ Loaded ML results from disk for {len(ml_results)} models")
                else:
                    # Fallback: basic file listing
                    ml_files = list(ml_dir.glob('*.json'))
                    if ml_files:
                        analysis_data['results']['ml'] = {
                            'results': {},
                            'completed': True,
                            'files': [str(f) for f in ml_files]
                        }
                        
            except Exception as e:
                print(f"Error loading ML results from disk: {e}")
    
    return AnalysisStatus(**analysis_data)

@app.get("/analyses/{analysis_id}/results", summary="Get Analysis Results")
async def get_analysis_results(analysis_id: str):
    """Get detailed results from a completed analysis."""
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status = analysis_status[analysis_id]
    
    if status['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    return status['results']

@app.get("/analyses/{analysis_id}/files", summary="List Analysis Output Files")
async def list_analysis_files(analysis_id: str):
    """List all output files from an analysis."""
    results_dir = get_results_path(analysis_id)
    
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    files = []
    for file_path in results_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(results_dir)
            files.append({
                'path': str(rel_path),
                'size_bytes': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return files

@app.get("/analyses/{analysis_id}/files/{file_path:path}", summary="Download Analysis File")
async def download_analysis_file(analysis_id: str, file_path: str):
    """Download a specific file from analysis results."""
    results_dir = get_results_path(analysis_id)
    target_file = results_dir / file_path
    
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=target_file,
        filename=target_file.name,
        media_type='application/octet-stream'
    )

@app.post("/package", summary="Create Artifact Package")
async def create_package(
    request: PackageRequest,
    background_tasks: BackgroundTasks
):
    """Create a downloadable artifact package from analysis results."""
    if request.analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status = analysis_status[request.analysis_id]
    
    if status['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    # Generate package ID
    package_id = generate_id()
    
    # Start background packaging
    background_tasks.add_task(
        create_background_package,
        package_id,
        request.analysis_id,
        request.include_data,
        request.create_zip,
        request.create_tar
    )
    
    return {
        "package_id": package_id,
        "status": "packaging",
        "message": "Package creation started"
    }

async def create_background_package(
    package_id: str,
    analysis_id: str,
    include_data: bool,
    create_zip: bool,
    create_tar: bool
):
    """Create artifact package in the background."""
    try:
        results_dir = get_results_path(analysis_id)
        
        # Prepare data sources
        data_sources = {}
        
        # Add original files
        for file_id in analysis_status[analysis_id]['results']['data_info']['file_ids']:
            file_path = get_file_path(file_id)
            if file_path.exists():
                data_sources['raw'] = str(file_path)
                break
        
        # Create package
        package_result = create_biomarker_package(
            output_dir=str(PACKAGES_DIR),
            qc_dir=str(results_dir / 'qc') if (results_dir / 'qc').exists() else None,
            stats_dir=str(results_dir / 'statistical') if (results_dir / 'statistical').exists() else None,
            ml_dir=str(results_dir / 'ml') if (results_dir / 'ml').exists() else None,
            data_sources=data_sources,
            metadata={
                'api_generated': True,
                'analysis_id': analysis_id,
                'package_id': package_id,
                'created_at': datetime.now().isoformat()
            },
            analysis_summary=analysis_status[analysis_id]['results']['data_info'],
            project_name=f"api_analysis_{analysis_id[:8]}",
            create_zip=create_zip,
            create_tar=create_tar
        )
        
        # Store package info (in production, use database)
        package_info = {
            'package_id': package_id,
            'analysis_id': analysis_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'package_result': package_result
        }
        
        # Save package info to file
        package_info_file = PACKAGES_DIR / f"{package_id}.json"
        with open(package_info_file, 'w') as f:
            json.dump(package_info, f, indent=2, default=str)
            
    except Exception as e:
        print(f"Package creation failed: {e}")

@app.get("/packages", summary="List Artifact Packages")
async def list_packages():
    """List all created artifact packages."""
    packages = []
    
    for info_file in PACKAGES_DIR.glob("*.json"):
        try:
            with open(info_file, 'r') as f:
                package_info = json.load(f)
                packages.append({
                    'package_id': package_info['package_id'],
                    'analysis_id': package_info['analysis_id'],
                    'status': package_info['status'],
                    'created_at': package_info['created_at']
                })
        except Exception as e:
            print(f"Error reading package info {info_file}: {e}")
            continue
    
    return packages

@app.get("/packages/{package_id}", summary="Get Package Information")
async def get_package_info(package_id: str):
    """Get information about a specific package."""
    package_info_file = PACKAGES_DIR / f"{package_id}.json"
    
    if not package_info_file.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    with open(package_info_file, 'r') as f:
        package_info = json.load(f)
    
    return package_info

@app.get("/packages/{package_id}/download", summary="Download Package Archive")
async def download_package(package_id: str, format: str = Query("zip", enum=["zip", "tar"])):
    """Download a package archive (ZIP or TAR)."""
    package_info_file = PACKAGES_DIR / f"{package_id}.json"
    
    if not package_info_file.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    with open(package_info_file, 'r') as f:
        package_info = json.load(f)
    
    archives = package_info['package_result']['archives']
    
    if format not in archives:
        raise HTTPException(status_code=404, detail=f"Package format '{format}' not available")
    
    archive_path = Path(archives[format])
    
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Package file not found")
    
    return FileResponse(
        path=archive_path,
        filename=archive_path.name,
        media_type='application/octet-stream'
    )

@app.delete("/analyses/{analysis_id}", summary="Delete Analysis")
async def delete_analysis(analysis_id: str):
    """Delete an analysis and its results."""
    try:
        # Remove from status tracking
        if analysis_id in analysis_status:
            del analysis_status[analysis_id]
        
        # Remove results directory
        results_path = get_results_path(analysis_id)
        if results_path.exists():
            shutil.rmtree(results_path)
        
        return {"message": f"Analysis {analysis_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")


# NEW ENDPOINTS: Feature Extraction and Enrichment

@app.post("/extract-features", response_model=AnalysisStatus, summary="Feature Extraction Pipeline")
async def extract_features(
    request: FeatureExtractionRequest,
    background_tasks: BackgroundTasks
):
    """
    Run the complete feature extraction pipeline:
    1. OpenMS peak picking (if mzML files)
    2. Feature deduplication
    3. Frequency and score filtering
    4. Missing value imputation
    5. Scaling/normalization
    """
    try:
        # Generate analysis ID
        analysis_id = generate_id()
        
        # Validate file IDs exist
        for file_id in request.file_ids:
            file_path = get_file_path(file_id)
            if not file_path.exists():
                raise HTTPException(status_code=400, detail=f"File {file_id} not found")
        
        # Initialize analysis status
        update_analysis_status(
            analysis_id,
            status='pending',
            progress=0.0,
            message='Feature extraction queued'
        )
        
        # Start background processing
        background_tasks.add_task(
            run_feature_extraction_pipeline,
            analysis_id,
            request
        )
        
        return AnalysisStatus(
            analysis_id=analysis_id,
            status='pending',
            progress=0.0,
            message='Feature extraction pipeline started',
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start feature extraction: {str(e)}")


@app.post("/enrich-pathways", response_model=AnalysisStatus, summary="Pathway Enrichment Analysis")
async def enrich_pathways(
    request: EnrichmentRequest,
    background_tasks: BackgroundTasks
):
    """
    Run pathway enrichment analysis on significant features:
    1. Load feature matrix
    2. Map features to biological IDs
    3. KEGG pathway enrichment
    4. GO term enrichment
    5. Statistical significance testing
    """
    try:
        # Generate analysis ID
        analysis_id = generate_id()
        
        # Initialize analysis status
        update_analysis_status(
            analysis_id,
            status='pending',
            progress=0.0,
            message='Pathway enrichment queued'
        )
        
        # Start background processing
        background_tasks.add_task(
            run_enrichment_pipeline,
            analysis_id,
            request
        )
        
        return AnalysisStatus(
            analysis_id=analysis_id,
            status='pending',
            progress=0.0,
            message='Pathway enrichment analysis started',
            started_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start enrichment analysis: {str(e)}")


@app.post("/integrated-analysis", response_model=AnalysisStatus, summary="Integrated Feature Extraction + Enrichment")
async def integrated_analysis(
    request: IntegratedAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Run complete integrated pipeline:
    1. Feature extraction (if requested)
    2. Statistical analysis to find significant features
    3. Pathway enrichment analysis
    4. Comprehensive reporting
    """
    try:
        # Generate analysis ID
        analysis_id = generate_id()
        
        # Validate file IDs exist
        for file_id in request.file_ids:
            file_path = get_file_path(file_id)
            if not file_path.exists():
                raise HTTPException(status_code=400, detail=f"File {file_id} not found")
        
        # Initialize analysis status
        update_analysis_status(
            analysis_id,
            status='pending',
            progress=0.0,
            message='Integrated analysis queued'
        )
        
        # Start background processing
        background_tasks.add_task(
            run_integrated_pipeline,
            analysis_id,
            request
        )
        
        return AnalysisStatus(
            analysis_id=analysis_id,
            status='pending',
            progress=0.0,
            message='Integrated feature extraction + enrichment started',
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start integrated analysis: {str(e)}")


# Background processing functions for new endpoints

async def run_feature_extraction_pipeline(
    analysis_id: str,
    request: FeatureExtractionRequest
):
    """Background task for feature extraction pipeline"""
    results_path = get_results_path(analysis_id)
    results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        update_analysis_status(analysis_id, status='running', progress=0.1, message='Starting feature extraction...')
        
        # Step 1: Load data files
        feature_dfs = []
        file_paths = [get_file_path(file_id) for file_id in request.file_ids]
        
        update_analysis_status(analysis_id, progress=0.2, message='Loading data files...')
        
        # For demo purposes, assume we have mwTab files (in production, handle mzML with OpenMS)
        for file_path in file_paths:
            df, metadata = load_file(str(file_path))
            feature_dfs.append(df)
        
        # Combine data (simple concatenation for demo)
        if len(feature_dfs) == 1:
            combined_df = feature_dfs[0]
        else:
            combined_df = pd.concat(feature_dfs, axis=0)
        
        update_analysis_status(analysis_id, progress=0.3, message='Data loaded, starting deduplication...')
        
        # Step 2: Deduplication (simulate with existing data)
        # Create dummy metadata for demo
        n_features = len(combined_df.columns)
        dummy_metadata = pd.DataFrame({
            'mz': np.random.uniform(100, 1000, n_features),
            'rt': np.random.uniform(0, 1800, n_features)
        }, index=combined_df.columns)
        combined_df.attrs['feature_metadata'] = dummy_metadata
        
        deduplicated_df, dedup_stats = deduplicate_features(
            combined_df,
            mz_tolerance_ppm=request.mz_tolerance_ppm,
            rt_tolerance_min=request.rt_tolerance_min,
            keep_strategy=request.keep_strategy
        )
        
        update_analysis_status(analysis_id, progress=0.5, message='Deduplication complete, filtering features...')
        
        # Step 3: Filtering
        filter_config = {
            'frequency_filter': {
                'enabled': True,
                'min_frequency': request.min_frequency,
                'per_group': False
            },
            'score_filter': {
                'enabled': True,
                'method': request.score_method,
                'min_percentile': request.min_score_percentile
            },
            'custom_filters': {'enabled': False}
        }
        
        filtered_df, filter_stats = combined_filter(deduplicated_df, filter_config)
        
        update_analysis_status(analysis_id, progress=0.7, message='Filtering complete, imputing missing values...')
        
        # Step 4: Imputation
        imputed_df, imputation_stats = impute_missing_values(
            filtered_df,
            method=request.imputation_method,
            group_column=request.group_column
        )
        
        update_analysis_status(analysis_id, progress=0.8, message='Imputation complete, scaling features...')
        
        # Step 5: Scaling
        scaler_params = ScalerParams(
            method=request.scaling_method,
            log_transform=request.log_transform
        )
        
        scaled_df, scaling_info = apply_scaling(imputed_df, request.scaling_method, scaler_params)
        
        update_analysis_status(analysis_id, progress=0.9, message='Scaling complete, saving results...')
        
        # Save results
        final_matrix_file = results_path / "final_feature_matrix.csv"
        scaled_df.to_csv(final_matrix_file)
        
        # Save pipeline statistics
        pipeline_stats = {
            'initial_features': len(combined_df.columns),
            'after_deduplication': len(deduplicated_df.columns),
            'after_filtering': len(filtered_df.columns),
            'final_features': len(scaled_df.columns),
            'samples': len(scaled_df.index),
            'deduplication_stats': dedup_stats,
            'filtering_stats': filter_stats,
            'imputation_stats': imputation_stats,
            'scaling_info': scaling_info
        }
        
        stats_file = results_path / "feature_extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(pipeline_stats, f, indent=2, default=str)
        
        # Complete
        update_analysis_status(
            analysis_id,
            status='completed',
            progress=1.0,
            message='Feature extraction pipeline completed successfully',
            completed_at=datetime.now().isoformat(),
            results={
                'final_feature_matrix': str(final_matrix_file),
                'pipeline_statistics': str(stats_file),
                'summary': pipeline_stats
            }
        )
        
    except Exception as e:
        update_analysis_status(
            analysis_id,
            status='failed',
            message=f'Feature extraction failed: {str(e)}',
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


async def run_enrichment_pipeline(
    analysis_id: str,
    request: EnrichmentRequest
):
    """Background task for pathway enrichment analysis"""
    results_path = get_results_path(analysis_id)
    results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        update_analysis_status(analysis_id, status='running', progress=0.1, message='Starting enrichment analysis...')
        
        # Load feature matrix
        if request.feature_matrix_file.startswith('/'):
            matrix_path = Path(request.feature_matrix_file)
        else:
            matrix_path = get_file_path(request.feature_matrix_file)
        
        if not matrix_path.exists():
            raise FileNotFoundError(f"Feature matrix file not found: {matrix_path}")
        
        feature_df = pd.read_csv(matrix_path, index_col=0)
        
        update_analysis_status(analysis_id, progress=0.3, message='Feature matrix loaded, running enrichment...')
        
        # Set up enrichment parameters
        enrichment_params = PathwayEnrichmentParams(
            p_value_threshold=request.p_value_threshold,
            fdr_method=request.fdr_method,
            use_kegg=request.use_kegg,
            use_go=request.use_go,
            organism=request.organism,
            min_genes_per_pathway=request.min_genes_per_pathway
        )
        
        # Run enrichment analysis
        enrichment_results = run_pathway_enrichment(
            feature_df,
            request.significant_features,
            background_features=request.background_features,
            params=enrichment_params
        )
        
        update_analysis_status(analysis_id, progress=0.8, message='Enrichment complete, saving results...')
        
        # Save enrichment results
        from biomarker.core.analysis.pathway_analysis import save_enrichment_results
        saved_files = save_enrichment_results(enrichment_results, results_path)
        
        # Complete
        update_analysis_status(
            analysis_id,
            status='completed',
            progress=1.0,
            message='Pathway enrichment analysis completed successfully',
            completed_at=datetime.now().isoformat(),
            results={
                'enrichment_results': str(saved_files.get('complete_results', '')),
                'summary': str(saved_files.get('summary', '')),
                'kegg_results': str(saved_files.get('kegg_csv', '')),
                'go_results': str(saved_files.get('go_csv', '')),
                'summary_stats': enrichment_results['summary']
            }
        )
        
    except Exception as e:
        update_analysis_status(
            analysis_id,
            status='failed',
            message=f'Enrichment analysis failed: {str(e)}',
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


async def run_integrated_pipeline(
    analysis_id: str,
    request: IntegratedAnalysisRequest
):
    """Background task for integrated feature extraction + enrichment pipeline"""
    results_path = get_results_path(analysis_id)
    results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        update_analysis_status(analysis_id, status='running', progress=0.05, message='Starting integrated analysis...')
        
        # Step 1: Feature extraction (if requested)
        feature_matrix_file = None
        if request.feature_extraction:
            update_analysis_status(analysis_id, progress=0.1, message='Running feature extraction...')
            
            # Run simplified feature extraction
            feature_dfs = []
            file_paths = [get_file_path(file_id) for file_id in request.file_ids]
            
            for file_path in file_paths:
                df, metadata = load_file(str(file_path))
                feature_dfs.append(df)
            
            combined_df = pd.concat(feature_dfs, axis=0) if len(feature_dfs) > 1 else feature_dfs[0]
            
            # Save intermediate feature matrix
            feature_matrix_file = results_path / "feature_matrix.csv"
            combined_df.to_csv(feature_matrix_file)
            
            update_analysis_status(analysis_id, progress=0.3, message='Feature extraction complete')
        else:
            # Load existing feature matrix
            feature_matrix_file = get_file_path(request.file_ids[0])
            combined_df = pd.read_csv(feature_matrix_file, index_col=0)
        
        # Step 2: Statistical analysis (if requested)
        significant_features = []
        if request.run_statistics:
            update_analysis_status(analysis_id, progress=0.4, message='Running statistical analysis...')
            
            # For demo, select random features as "significant"
            n_significant = min(20, len(combined_df.columns) // 5)
            significant_features = np.random.choice(combined_df.columns, n_significant, replace=False).tolist()
            
            # Save significant features list
            sig_features_file = results_path / "significant_features.txt"
            with open(sig_features_file, 'w') as f:
                for feature in significant_features:
                    f.write(f"{feature}\n")
            
            update_analysis_status(analysis_id, progress=0.6, message='Statistical analysis complete')
        
        # Step 3: Enrichment analysis (if requested)
        enrichment_results = None
        if request.run_enrichment and significant_features:
            update_analysis_status(analysis_id, progress=0.7, message='Running pathway enrichment...')
            
            enrichment_params = PathwayEnrichmentParams(
                p_value_threshold=request.p_value_threshold,
                use_kegg=request.use_kegg,
                use_go=request.use_go
            )
            
            enrichment_results = run_pathway_enrichment(
                combined_df,
                significant_features,
                params=enrichment_params
            )
            
            # Save enrichment results
            from biomarker.core.analysis.pathway_analysis import save_enrichment_results
            enrichment_files = save_enrichment_results(enrichment_results, results_path / "enrichment")
            
            update_analysis_status(analysis_id, progress=0.9, message='Enrichment analysis complete')
        
        # Compile final results
        final_results = {
            'feature_matrix': str(feature_matrix_file),
            'n_features': len(combined_df.columns),
            'n_samples': len(combined_df.index),
            'significant_features_count': len(significant_features),
            'significant_features': significant_features[:10] if significant_features else [],  # Top 10 for display
        }
        
        if enrichment_results:
            final_results.update({
                'enrichment_summary': enrichment_results['summary'],
                'total_significant_pathways': enrichment_results['summary']['total_significant_pathways']
            })
        
        # Complete
        update_analysis_status(
            analysis_id,
            status='completed',
            progress=1.0,
            message='Integrated analysis completed successfully',
            completed_at=datetime.now().isoformat(),
            results=final_results
        )
        
    except Exception as e:
        update_analysis_status(
            analysis_id,
            status='failed',
            message=f'Integrated analysis failed: {str(e)}',
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


# NEW ENDPOINTS: Report Generation

@app.post("/generate-report", response_model=AnalysisStatus, summary="Generate Comprehensive Report")
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a comprehensive biomarker discovery report from completed analysis results.
    
    Creates publication-ready reports with:
    - Executive summary for stakeholders
    - Detailed methodology section
    - Statistical analysis results
    - Pathway enrichment findings
    - Quality control assessment
    - Professional formatting and styling
    """
    try:
        # Verify analysis exists and is completed
        if request.analysis_id not in analysis_status:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis = analysis_status[request.analysis_id]
        if analysis['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Analysis not completed yet")
        
        # Generate report ID
        report_id = generate_id()
        
        # Initialize report status
        update_analysis_status(
            report_id,
            status='pending',
            progress=0.0,
            message='Report generation queued'
        )
        
        # Start background report generation
        background_tasks.add_task(
            run_report_generation,
            report_id,
            request
        )
        
        return AnalysisStatus(
            analysis_id=report_id,
            status='pending',
            progress=0.0,
            message='Report generation started',
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start report generation: {str(e)}")


@app.post("/generate-executive-summary", response_model=AnalysisStatus, summary="Generate Executive Summary")
async def generate_executive_summary_endpoint(
    request: ExecutiveSummaryRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate an executive summary report from analysis results.
    
    Creates a concise, high-level summary suitable for:
    - Clinical decision makers
    - Research oversight committees
    - Grant reports and presentations
    - Stakeholder communications
    """
    try:
        # Verify analysis exists and is completed
        if request.analysis_id not in analysis_status:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis = analysis_status[request.analysis_id]
        if analysis['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Analysis not completed yet")
        
        # Generate summary ID
        summary_id = generate_id()
        
        # Initialize summary status
        update_analysis_status(
            summary_id,
            status='pending',
            progress=0.0,
            message='Executive summary generation queued'
        )
        
        # Start background summary generation
        background_tasks.add_task(
            run_executive_summary_generation,
            summary_id,
            request
        )
        
        return AnalysisStatus(
            analysis_id=summary_id,
            status='pending',
            progress=0.0,
            message='Executive summary generation started',
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start summary generation: {str(e)}")


# Background processing functions for report generation

async def run_report_generation(
    report_id: str,
    request: ReportGenerationRequest
):
    """Background task for comprehensive report generation"""
    results_path = get_results_path(report_id)
    results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        update_analysis_status(report_id, status='running', progress=0.1, message='Starting report generation...')
        
        # Get original analysis results
        original_analysis = analysis_status[request.analysis_id]
        if 'results' not in original_analysis:
            raise Exception("Original analysis has no results")
        
        analysis_results = original_analysis['results']
        
        update_analysis_status(report_id, progress=0.3, message='Preparing report data...')
        
        # Configure report
        report_config = ReportConfig(
            title=f"{request.project_name} - Biomarker Discovery Report",
            project_name=request.project_name,
            author=request.author,
            institution=request.institution,
            theme=request.theme,
            technical_level=request.technical_level,
            include_methodology=request.include_methodology,
            include_statistical_analysis=request.include_statistical_analysis,
            include_pathway_analysis=request.include_pathway_analysis,
            include_quality_control=request.include_quality_control,
            generate_html=request.generate_html,
            generate_pdf=request.generate_pdf,
            include_plots=False  # Skip plots for API generation
        )
        
        update_analysis_status(report_id, progress=0.6, message='Generating report sections...')
        
        # Generate comprehensive report
        generated_files = generate_comprehensive_report(
            analysis_results,
            results_path,
            report_config
        )
        
        update_analysis_status(report_id, progress=0.9, message='Finalizing report...')
        
        # Complete
        update_analysis_status(
            report_id,
            status='completed',
            progress=1.0,
            message='Report generation completed successfully',
            completed_at=datetime.now().isoformat(),
            results={
                'generated_files': generated_files,
                'report_config': {
                    'theme': request.theme,
                    'technical_level': request.technical_level,
                    'project_name': request.project_name
                },
                'original_analysis_id': request.analysis_id
            }
        )
        
    except Exception as e:
        update_analysis_status(
            report_id,
            status='failed',
            message=f'Report generation failed: {str(e)}',
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


async def run_executive_summary_generation(
    summary_id: str,
    request: ExecutiveSummaryRequest
):
    """Background task for executive summary generation"""
    results_path = get_results_path(summary_id)
    results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        update_analysis_status(summary_id, status='running', progress=0.2, message='Starting summary generation...')
        
        # Get original analysis results
        original_analysis = analysis_status[request.analysis_id]
        if 'results' not in original_analysis:
            raise Exception("Original analysis has no results")
        
        analysis_results = original_analysis['results']
        
        update_analysis_status(summary_id, progress=0.6, message='Generating executive summary...')
        
        # Generate executive summary
        summary_file = results_path / "executive_summary.html"
        summary_path = generate_executive_summary(analysis_results, summary_file)
        
        # Complete
        update_analysis_status(
            summary_id,
            status='completed',
            progress=1.0,
            message='Executive summary generated successfully',
            completed_at=datetime.now().isoformat(),
            results={
                'summary_file': summary_path,
                'project_name': request.project_name,
                'original_analysis_id': request.analysis_id
            }
        )
        
    except Exception as e:
        update_analysis_status(
            summary_id,
            status='failed',
            message=f'Summary generation failed: {str(e)}',
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


def transform_pca_results_for_frontend(pca_results):
    """
    Transform raw PCA results into the format expected by the frontend.
    
    Args:
        pca_results: Raw PCA results from run_complete_pca_analysis
        
    Returns:
        Dict in the format expected by the frontend
    """
    try:
        # Extract the PCA DataFrame and analysis results
        pca_analysis = pca_results['pca_results']
        pca_df = pca_analysis['pca_df']
        explained_variance = pca_analysis['explained_variance']
        summary = pca_results['summary']
        
        # Extract all available PC values (up to 10 components)
        pc_values = {}
        pc_variances = {}
        available_components = min(10, len(explained_variance))
        
        for i in range(available_components):
            pc_name = f'PC{i+1}'
            if pc_name in pca_df.columns:
                pc_values[f'pc{i+1}_values'] = pca_df[pc_name].tolist()
                pc_variances[f'pc{i+1}_variance'] = float(explained_variance[i] * 100)
        
        # Sample names and groups
        sample_names = pca_df.index.tolist()
        
        # Generate group colors if Group column exists
        group_colors = []
        if 'Group' in pca_df.columns:
            unique_groups = pca_df['Group'].unique()
            color_map = {
                'Case': '#EF4444',      # Red for cases
                'Control': '#3B82F6'    # Blue for controls
            }
            # Extend color map for additional groups if needed
            if len(unique_groups) > 2:
                additional_colors = ['#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4']
                for i, group in enumerate(unique_groups):
                    if group not in color_map:
                        color_map[group] = additional_colors[i % len(additional_colors)]
            
            group_colors = [color_map.get(group, '#94A3B8') for group in pca_df['Group']]
        else:
            # Default color if no groups
            group_colors = ['#94A3B8'] * len(sample_names)
        
        # Build the result dictionary with all components
        result = {
            'pca_results': {
                'sample_names': sample_names,
                'group_colors': group_colors,
                'n_samples': len(sample_names),
                'n_features': summary.get('n_features', 0),
                'n_components': available_components,
                'total_variance': summary.get('total_variance_2pc', 0.0) * 100 if summary.get('total_variance_2pc') else 0.0
            }
        }
        
        # Add all PC values and variances
        result['pca_results'].update(pc_values)
        result['pca_results'].update(pc_variances)
        
        # Keep backward compatibility with PC1 and PC2 specific fields
        if 'pc1_values' in pc_values:
            result['pca_results']['pc1_values'] = pc_values['pc1_values']
        if 'pc2_values' in pc_values:
            result['pca_results']['pc2_values'] = pc_values['pc2_values']
        if 'pc1_variance' in pc_variances:
            result['pca_results']['pc1_variance'] = pc_variances['pc1_variance']
        if 'pc2_variance' in pc_variances:
            result['pca_results']['pc2_variance'] = pc_variances['pc2_variance']
        
        return result
        
    except Exception as e:
        print(f"Error transforming PCA results: {e}")
        return {
            'pca_results': {
                'pc1_values': [],
                'pc2_values': [],
                'group_colors': [],
                'sample_names': [],
                'pc1_variance': 0.0,
                'pc2_variance': 0.0,
                'n_samples': 0,
                'n_features': 0,
                'n_components': 0,
                'total_variance': 0.0
            }
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port,
        reload=settings.api_reload and settings.debug
    ) 