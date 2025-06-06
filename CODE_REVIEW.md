# OmniBio Code Review - mzML Integration

## Backend Findings

- `file_loader.detect_file_type` correctly distinguishes mwTab vs mzML profile vs centroid by inspecting first few spectra.
- `extract_ms1_data_from_mzml` handles chromatogram data or falls back to MS1 binning; however, file detection may mislabel some centroid files due to limited heuristics.
- `batch_mzml_processor.process_mzml_batch` successfully aggregates multiple mzML files using this loader but lacks peak centroiding step.
- `api/main.py` exposes endpoints for feature extraction and analysis but examples still assume mwTab files in some comments.  Endpoint `/extract-features` should be tested with real mzML uploads.
- `peak_picking.run_openms_feature_finder` wraps OpenMS FeatureFinderMetabo but requires `pyopenms`; missing dependency will throw `FeatureExtractionError`.

### Potential Improvements

1. **Centroid Conversion** – implement Task A.2 using `pyopenms.PeakPickerHiRes` so that profile mzML files can be automatically centroided before feature extraction.
2. **Enhanced File Detection** – check `file_loader.detect_file_type` for more robust mzML profile vs centroid detection, perhaps by reading instrument parameters.
3. **Error Handling** – `test_scaling_integration.py` fails when API server is not running. Add try/except to give a clearer message.
4. **Database warnings** – as noted in `ERROR_ANALYSIS.md`, some columns such as `file_metadata` are missing. Update migrations to avoid runtime warnings.
5. **Missing Dependencies** – docs highlight optional packages `pymzml` and `lightgbm`. Ensure `requirements.txt` lists them (already included) and document their installation.

## Frontend Findings

- Dashboard (Next.js) polls `/analyses` every 5 s. When no backend is running, users receive a generic "Failed to load data"; consider handling connection errors more gracefully.
- `NewAnalysisModal` currently only allows selecting one analysis type at a time but server supports combined lists; UI already maps "Full" to `["qc","pca","statistical","ml"]`.
- `UploadDropzone` and `FileTable` treat file types generically; ensure `.mzML` extension is accepted.
- No dedicated view for QC or PCA plots yet; results page only links to job details.

### Potential Improvements

1. **Loading States** – show spinner for file list and job list while fetching.
2. **Error Boundaries** – wrap API interactions in error boundaries to avoid white screens (see `troubleshoot_frontend.md`).
3. **mzML Visualization** – add small previews of TIC/BPC PNGs once backend generates them.

## Additional Features for Case‑Control Analysis

- **Batch Effect Exploration** – implement PCA plot generation on mzML feature matrix (Task A.4) to visualise clustering of cases vs controls.
- **Duplicate Removal & Frequency Filtering** – implement Tasks B.6 and B.7 to clean feature matrix before ML.
- **Missing Value Imputation** – Task B.8 ensures robust statistical analysis.
- **Model Training** – logistic regression and LightGBM with cross‑validation (Task C.10).
- **Reporting** – packaging of results and vector database for RAG-based summaries (Tasks D.12‑D.14).

Testing `test_batch_mzml.py` on the supplied data produces a combined mwTab file with 5512 features and confirms direct mzML upload works:
```
📊 Results Summary:
   Samples: 10
   Features: 5512
   Output file: mzml_test_output/lipidomics_batch_test_improved.txt
```
Additional scaling integration tests require the API server running; otherwise they fail with connection errors.
