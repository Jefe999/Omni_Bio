# OmniBio Biomarker Discovery API

## Overview

The OmniBio Biomarker Discovery API provides REST endpoints for biomarker discovery from metabolomics data. This FastAPI-based service supports file upload, analysis execution, results retrieval, and artifact packaging.

## Features

- **File Upload**: Support for mwTab and mzML formats with auto-detection
- **Analysis Pipeline**: Quality control, statistical analysis, and machine learning
- **Background Processing**: Async analysis execution with progress tracking
- **Results Management**: Download individual files or complete analysis packages
- **Artifact Packaging**: Professional packages with HTML reports and documentation

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Ensure biomarker pipeline dependencies are installed
pip install pandas numpy matplotlib scikit-learn
```

### Running the Server

```bash
# Start the API server
cd biomarker/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Server will be available at:
# - API: http://localhost:8000
# - Interactive docs: http://localhost:8000/docs
# - ReDoc docs: http://localhost:8000/redoc
```

## API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Basic health check |
| GET | `/health` | Detailed system status |

### File Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload data file (mwTab/mzML) |
| GET | `/files` | List all uploaded files |
| DELETE | `/files/{file_id}` | Delete uploaded file |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Start biomarker analysis |
| GET | `/analyses` | List all analyses |
| GET | `/analyses/{analysis_id}` | Get analysis status |
| GET | `/analyses/{analysis_id}/results` | Get analysis results |
| GET | `/analyses/{analysis_id}/files` | List output files |
| GET | `/analyses/{analysis_id}/files/{path}` | Download output file |
| DELETE | `/analyses/{analysis_id}` | Delete analysis |

### Artifact Packaging

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/package` | Create artifact package |
| GET | `/packages` | List all packages |
| GET | `/packages/{package_id}` | Get package info |
| GET | `/packages/{package_id}/download` | Download package |

## Usage Examples

### 1. Upload File

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.txt"
```

Response:
```json
{
  "file_id": "abc123...",
  "filename": "data.txt",
  "file_type": "mwtab",
  "size_bytes": 1024000,
  "uploaded_at": "2024-01-15T10:00:00",
  "status": "uploaded"
}
```

### 2. Start Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_ids": ["abc123..."],
    "analysis_types": ["qc", "statistical", "ml"],
    "project_name": "my_analysis"
  }'
```

### 3. Check Analysis Status

```bash
curl "http://localhost:8000/analyses/xyz789..."
```

Response:
```json
{
  "analysis_id": "xyz789...",
  "status": "completed",
  "progress": 1.0,
  "message": "Analysis completed successfully",
  "started_at": "2024-01-15T10:05:00",
  "completed_at": "2024-01-15T10:07:30",
  "results": {
    "data_info": {
      "n_samples": 199,
      "n_features": 727
    }
  }
}
```

### 4. Create Package

```bash
curl -X POST "http://localhost:8000/package" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "xyz789...",
    "create_zip": true
  }'
```

### 5. Download Package

```bash
curl -O "http://localhost:8000/packages/pkg123.../download?format=zip"
```

## Data Models

### AnalysisRequest

```python
{
  "file_ids": ["string"],           # Required: List of uploaded file IDs
  "analysis_types": ["string"],     # Optional: ["qc", "statistical", "ml", "pca"]
  "project_name": "string",         # Optional: Analysis project name
  "group_column": "string"          # Optional: Column for sample groups
}
```

### AnalysisStatus

```python
{
  "analysis_id": "string",
  "status": "string",               # pending, running, completed, failed
  "progress": 0.0,                  # 0.0 to 1.0
  "message": "string",
  "started_at": "datetime",
  "completed_at": "datetime",       # null if not completed
  "results": {...},                 # null if not completed
  "error": "string"                 # null if no error
}
```

## Analysis Types

| Type | Description | Outputs |
|------|-------------|---------|
| `qc` | Quality control plots | TIC/BPC chromatograms |
| `pca` | Principal component analysis | PCA plots, loadings, batch effect detection |
| `statistical` | Statistical tests | T-tests, volcano plots |
| `ml` | Machine learning models | ROC curves, feature importance |

**Note**: PCA analysis is fully functional as of v1.0.0 with plotly 6.1.2+ and NumPy 2.x compatibility.

## File Format Support

| Format | Extension | Auto-Detection | Description |
|--------|-----------|----------------|-------------|
| mwTab | `.txt` | ✅ | Metabolomics Workbench format |
| mzML | `.mzML` | ✅ | Mass spectrometry data |

## Error Handling

The API uses standard HTTP status codes:

- **200**: Success
- **400**: Bad request (validation error)
- **404**: Resource not found
- **422**: Unprocessable entity (data validation error)
- **500**: Internal server error

Error responses include detailed messages:

```json
{
  "detail": "Analysis not found"
}
```

## Background Processing

Long-running analyses are processed in the background:

1. **Submit Analysis**: Returns immediately with `analysis_id`
2. **Monitor Progress**: Poll status endpoint for updates
3. **Retrieve Results**: Access results when status is `completed`

## Configuration

Environment variables:

- `UPLOAD_DIR`: Directory for uploaded files (default: `uploads/`)
- `RESULTS_DIR`: Directory for analysis results (default: `api_results/`)
- `PACKAGES_DIR`: Directory for artifact packages (default: `api_packages/`)

## Production Deployment

For production use:

1. **Database**: Replace in-memory storage with persistent database
2. **Authentication**: Add JWT or API key authentication
3. **Rate Limiting**: Implement request rate limiting
4. **File Storage**: Use cloud storage for large files
5. **Monitoring**: Add logging and health monitoring
6. **HTTPS**: Enable SSL/TLS encryption

## Interactive Documentation

When the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can test endpoints directly.

## Example Workflow

1. **Upload Data**: `POST /upload` with your metabolomics file
2. **Start Analysis**: `POST /analyze` with the file ID
3. **Monitor Progress**: `GET /analyses/{id}` until completed
4. **Review Results**: `GET /analyses/{id}/results` and download files
5. **Create Package**: `POST /package` for shareable artifacts
6. **Download Package**: `GET /packages/{id}/download` for final results

## Support

- Pipeline Documentation: See main README
- API Issues: Check server logs and error responses
- Development: Modify `main.py` for custom endpoints 