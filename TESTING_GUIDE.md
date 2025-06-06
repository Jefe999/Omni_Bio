# OmniBio Testing & Visualization Guide

## **🚀 Quick Start Testing**

### **Step 1: Start Both Services**
```bash
# Terminal 1 - Backend (in biomarker/api directory)
conda activate metabo
python -m uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend (in omnibio-frontend directory)  
npm run dev
```

### **Step 2: Access the Application**
1. **Frontend**: http://localhost:3000
2. **Backend API**: http://localhost:8000/docs (Swagger UI)
3. **Health Check**: http://localhost:8000/health

---

## **🔐 Authentication Testing**

### **Login Process**
1. Navigate to http://localhost:3000
2. Click **"API Key"** tab
3. Enter: `omnibio-dev-key-12345`
4. Click **"Authenticate"**

✅ **Expected**: Redirect to dashboard

---

## **📁 File Upload Testing**

### **For mwTab Files (.txt)**
**Strategy**: Upload individually or in small batches
- **Single File**: Best for testing/exploration
- **Multiple Files**: When comparing datasets

**Steps**:
1. Go to **"Files"** tab in dashboard
2. Drag & drop `ST002091_AN003415.txt` (provided sample)
3. **Expected**: Green checkmark, file appears in table

### **For mzML Files**
**Strategy**: Upload ALL mzML files as a BATCH
- **Why**: mzML files represent individual samples that should be analyzed together
- **Typical workflow**: 10-50 files representing different samples/conditions

**Steps**:
1. Select **multiple mzML files** at once (Ctrl+click or Cmd+click)
2. Drag all files to upload zone
3. **Expected**: All files upload with progress bar

### **File Types Supported**
- ✅ **mwTab** (.txt): Metabolomics Workbench format
- ✅ **mzML** (.mzml): Mass spectrometry data
- ✅ **JSON** (.json): Structured data
- ❌ Other formats will show validation errors (expected behavior)

---

## **🧪 Analysis Testing**

### **Create New Analysis**
1. Click **"New Analysis"** button
2. **Select Files**: Choose uploaded files
3. **Analysis Type**: Pick "Statistical" or "Machine Learning"  
4. **Project Name**: Enter descriptive name
5. Click **"Create Analysis"**

### **Monitor Progress**
- **Real-time updates**: Jobs refresh every 5 seconds
- **Status tracking**: "running" → "completed" or "failed"
- **Progress bar**: Shows 0-100% completion

### **Expected Timeline**
- **mwTab Analysis**: 10-30 seconds
- **mzML Analysis**: 2-5 minutes (depends on file size)

---

## **📊 Results Visualization**

### **Built-in Visualizations**
The system automatically generates:

#### **Statistical Analysis**
- **Volcano Plot**: `results/{analysis_id}/statistical/volcano_plot.png`
- **PCA Plots**: Principal component analysis
- **Box Plots**: Feature distributions
- **Heatmaps**: Correlation matrices

#### **Machine Learning**
- **ROC Curves**: Model performance
- **Feature Importance**: Top biomarkers
- **Confusion Matrix**: Classification results

### **Accessing Results**
1. **Via Frontend**: Click on completed job cards
2. **Via API**: GET `/analyses/{analysis_id}/results`
3. **File System**: Check `results/` directory

### **Download Options**
- **JSON**: Structured data for further analysis
- **CSV**: Spreadsheet-compatible
- **PNG**: Publication-ready plots

---

## **🔍 Manual Testing Workflow**

### **Test 1: Basic mwTab Analysis**
```bash
# 1. Upload sample file
curl -X POST "http://localhost:8000/upload" \
  -H "X-API-Key: omnibio-dev-key-12345" \
  -F "file=@ST002091_AN003415.txt"

# 2. Create analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: omnibio-dev-key-12345" \
  -d '{
    "file_ids": ["YOUR_FILE_ID_HERE"],
    "analysis_types": ["statistical"],
    "project_name": "Test Analysis"
  }'

# 3. Check results
curl -X GET "http://localhost:8000/analyses" \
  -H "X-API-Key: omnibio-dev-key-12345"
```

### **Test 2: Frontend Workflow**
1. **Login** → Dashboard loads
2. **Upload** → Files appear in table  
3. **Analyze** → Job starts running
4. **Monitor** → Progress updates
5. **Results** → View/download outputs

---

## **📈 Advanced Visualization**

### **Custom Analysis Script**
```python
# analyze_results.py
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

API_KEY = "omnibio-dev-key-12345"
BASE_URL = "http://localhost:8000"

def get_analysis_results(analysis_id):
    """Fetch and visualize analysis results"""
    headers = {"X-API-Key": API_KEY}
    
    # Get analysis info
    response = requests.get(f"{BASE_URL}/analyses/{analysis_id}", headers=headers)
    analysis = response.json()
    
    # Get results
    response = requests.get(f"{BASE_URL}/analyses/{analysis_id}/results", headers=headers)
    results = response.json()
    
    return analysis, results

def create_custom_plots(results):
    """Create custom visualizations"""
    if 'statistical_results' in results:
        stats = results['statistical_results']
        
        # Volcano plot data
        if 'significant_features' in stats:
            features = pd.DataFrame(stats['significant_features'])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(features['log2_fold_change'], -np.log10(features['p_value']))
            plt.xlabel('Log2 Fold Change')
            plt.ylabel('-Log10 P-value')
            plt.title('Volcano Plot')
            plt.savefig('custom_volcano.png', dpi=300, bbox_inches='tight')
            plt.show()

# Example usage
# analysis_id = "your-analysis-id-here"
# analysis, results = get_analysis_results(analysis_id)
# create_custom_plots(results)
```

---

## **🔧 Troubleshooting**

### **Common Issues & Solutions**

#### **Upload Fails**
- ✅ Check file format (.txt, .mzml, .json)
- ✅ Verify API key is correct
- ✅ Ensure file size < 100MB

#### **Analysis Stuck**
- ✅ Check backend logs for errors
- ✅ Verify files are valid format
- ✅ Restart backend if needed

#### **No Results**
- ✅ Wait for analysis completion
- ✅ Check analysis status via API
- ✅ Look in `results/` directory

### **Expected Warnings (Non-Breaking)**
```
Warning: pymzml not available. mzML support disabled.
LightGBM not available. Install with: pip install lightgbm
```
These are optional features - core functionality works fine.

---

## **🎯 Success Criteria**

### **System Working If**:
- ✅ Frontend loads without errors
- ✅ Login with API key succeeds  
- ✅ Files upload successfully
- ✅ Analysis jobs complete
- ✅ Results are generated

### **Expected Performance**:
- **mwTab (1MB)**: ~15 seconds
- **mzML batch (10 files)**: ~3 minutes
- **Real-time updates**: 5-second refresh

---

## **📊 Sample Results Structure**

```json
{
  "analysis_id": "abc123...",
  "status": "completed",
  "results": {
    "statistical_results": {
      "significant_features": [...],
      "volcano_plot_path": "results/.../volcano_plot.png",
      "summary": {
        "total_features": 682,
        "significant_raw": 238,
        "significant_adjusted": 3
      }
    },
    "plots": {
      "volcano": "path/to/volcano.png",
      "pca": "path/to/pca.png"
    }
  }
}
```

**Ready to start testing!** 🚀 