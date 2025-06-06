# OmniBio Biomarker Analysis - Agent Guide

## 🏗️ Project Overview

This is a full-stack biomarker discovery platform combining metabolomics data analysis with machine learning. The system processes mwTab and mzML files to identify significant metabolites and biomarkers for disease diagnosis.

### Architecture
```
├── biomarker/           # Python backend (FastAPI/Uvicorn)
│   ├── api/            # REST API endpoints
│   ├── core/           # ML pipeline & data processing
│   ├── io/             # File I/O utilities
│   └── models/         # Data models
├── omnibio-frontend/   # Next.js React frontend
│   ├── app/           # Next.js app router
│   ├── components/    # React components
│   └── lib/           # Utilities
└── config/            # Configuration files
```

## 🚀 Development Environment

### Backend Setup (Python)
```bash
# Navigate to backend
cd biomarker/api

# Activate conda environment
conda activate metabo

# Start backend server
python -m uvicorn main:app --reload --port 8000
```

### Frontend Setup (React/Next.js)
```bash
# Navigate to frontend
cd omnibio-frontend

# Install dependencies (use npm, not pnpm for this project)
npm install

# Start development server
npm run dev  # Runs on port 3000
```

### Port Management
- **Backend**: http://localhost:8000 (API + Swagger docs at /docs)
- **Frontend**: http://localhost:3000 (Next.js dev server)
- **Alternative Frontend**: http://localhost:3001 (if 3000 is occupied)

**Port Conflicts**: If servers won't start, clear ports first:
```bash
# Kill processes on specific ports
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
lsof -ti:3001 | xargs kill -9
```

## 📁 Key Files & Directories to Focus On

### Backend Core Files
- `biomarker/core/ml/model_pipeline.py` - **CRITICAL**: ML training & feature importance
- `biomarker/io/mwtab_parser.py` - mwTab file parsing & metabolite name extraction
- `biomarker/api/main.py` - FastAPI routes & endpoints
- `biomarker/core/processing/` - Data preprocessing pipelines

### Frontend Core Files
- `omnibio-frontend/app/` - Next.js pages & routing
- `omnibio-frontend/components/` - React components
- `omnibio-frontend/lib/api.ts` - Backend API integration

### Configuration & Data
- `config.yaml` - System configuration
- `docker-compose.yml` - Container setup
- `TESTING_GUIDE.md` - Comprehensive testing procedures

## 🧬 Metabolite Analysis Workflow

### Critical Concept: Feature Names vs Metabolite Names
**The core issue we're solving**: ML models show "Feature 48" instead of actual metabolite names like "L-Glutamine" or "Citric acid".

**Root Cause**: ML pipeline uses feature indices (0,1,2...) but doesn't properly map them to metabolite names from mwTab files.

**Solution**: In `model_pipeline.py`, the `save_model_artifacts` function now maps feature indices to actual metabolite names when available.

### Data Flow
1. **File Upload** → mwTab/mzML files uploaded via frontend
2. **Parsing** → `mwtab_parser.py` extracts metabolite names & stores in metadata
3. **ML Pipeline** → `model_pipeline.py` trains models with proper feature mapping
4. **Results** → Frontend displays actual metabolite names in feature importance

## 🧪 Testing & Validation

### Quick Development Test
```bash
# 1. Start both servers (see Dev Environment above)
# 2. Test backend health
curl http://localhost:8000/health
# 3. Test frontend (browser)
open http://localhost:3000
```

### Full Analysis Test
1. **Authentication**: Use API key `omnibio-dev-key-12345`
2. **Upload Test File**: Use provided `ST002091_AN003415.txt`
3. **Run Analysis**: Create ML analysis and verify metabolite names appear
4. **Check Results**: Ensure feature importance shows metabolite names, not "Feature X"

### Data Files for Testing
- `ST002091_AN003415.txt` - Sample mwTab file (NAFLD dataset)
- `nafld_dataset.txt` - Alternative test dataset
- `test_lipid_sample.mzML` - Sample mzML file

## 🛠️ Code Guidelines

### Python Backend Standards
- Use **type hints** for all function parameters and returns
- Add **logging** for debugging ML pipeline issues:
  ```python
  print(f"✓ Added metabolite names to {len(features)} features")
  ```
- **Error handling** with try/catch blocks for file operations
- Follow **FastAPI** patterns for API endpoints

### React Frontend Standards
- Use **TypeScript** for all components
- **Tailwind CSS** for styling (configured with shadcn/ui)
- **Server components** where possible (Next.js App Router)
- Handle **loading states** and **error boundaries**

### ML Pipeline Standards
- Always **map feature indices to metabolite names** when available
- **Save comprehensive metadata** in model artifacts
- **Log model performance metrics** for debugging
- **Validate input data** before training

## 🔧 Common Issues & Solutions

### Frontend Compilation Errors
```bash
# Missing react-plotly.js dependency
cd omnibio-frontend
npm install react-plotly.js plotly.js

# Component import issues
# Check imports in components/ui/ directory
```

### Backend ML Issues
```bash
# Feature name mapping problems
# Check logs in model_pipeline.py for:
# "✓ Added metabolite names to X features"

# Model training failures
# Verify mwTab parsing extracted metabolite names correctly
```

### Port/Server Issues
```bash
# Clear all ports and restart fresh
./kill_ports.sh  # (create if needed)
# Then restart both servers
```

## 📋 Agent Work Instructions

### When Working on ML Features
1. **Always test metabolite name mapping** - verify `feature_name` field appears in importance results
2. **Check both backend logs and frontend display** - ensure full pipeline works
3. **Use the test dataset** `ST002091_AN003415.txt` for consistent testing
4. **Clear ports between tests** to avoid stale server issues

### When Working on Frontend
1. **Test API integration first** - verify backend returns expected data structure
2. **Handle loading states** - analysis can take 30 seconds to 5 minutes
3. **Error boundaries** - gracefully handle API failures
4. **Responsive design** - use Tailwind classes for mobile compatibility

### When Working on Data Processing
1. **Preserve metabolite metadata** throughout the pipeline
2. **Validate file formats** before processing
3. **Log key steps** for debugging complex workflows
4. **Test with both mwTab and mzML formats**

## 🚦 Definition of Done

### Feature Complete When:
- [ ] Backend API returns proper metabolite names (not "Feature X")
- [ ] Frontend displays metabolite names correctly
- [ ] All existing tests pass
- [ ] New feature has test coverage
- [ ] Both servers start cleanly after port clearing
- [ ] Documentation updated if needed

### Testing Checklist:
- [ ] Backend health check passes
- [ ] File upload works (both formats)
- [ ] Analysis completes successfully
- [ ] Results show metabolite names
- [ ] Frontend handles loading/error states
- [ ] No console errors in browser

## 📚 Key Documentation
- `TESTING_GUIDE.md` - Comprehensive testing procedures
- `GRAPH_IMPROVEMENTS.md` - Visualization enhancements
- `ERROR_ANALYSIS.md` - Common error patterns
- `troubleshoot_frontend.md` - Frontend-specific issues

---
**Remember**: This is a biomarker discovery platform where accuracy of metabolite identification is critical for scientific validity. Always prioritize proper metabolite name mapping over generic feature labels. 