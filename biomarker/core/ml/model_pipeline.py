#!/usr/bin/env python3
"""
Model Training Pipeline for OmniBio MVP
Includes Logistic Regression and LightGBM with k-fold cross-validation
Based on existing biomarker_ml.py functionality
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
import warnings

# Optional LightGBM import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")


class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess DataFrame for ML - based on biomarker_metabo.py preprocess_df
    
    Args:
        df: Raw feature matrix DataFrame
        
    Returns:
        Preprocessed DataFrame ready for ML
    """
    print("Preprocessing data for ML...")
    
    # 1. Replace ±∞ with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Drop any feature that's 100% NaN up front
    initial_features = df.shape[1]
    df = df.dropna(axis=1, how='all')
    dropped_all_nan = initial_features - df.shape[1]
    if dropped_all_nan > 0:
        print(f"  - Dropped {dropped_all_nan} features with all NaN values")
    
    # 3. Drop extreme-TIC samples by IQR
    tic = df.sum(axis=1)
    q1, q3 = np.percentile(tic, [25, 75])
    iqr = q3 - q1
    mask = (tic >= q1 - 1.5*iqr) & (tic <= q3 + 1.5*iqr)
    initial_samples = df.shape[0]
    df = df.loc[mask]
    dropped_samples = initial_samples - df.shape[0]
    if dropped_samples > 0:
        print(f"  - Dropped {dropped_samples} outlier samples by TIC")
    
    # 4. Probabilistic Quotient Normalisation
    ref = df.median() + 1e-12  # Add small constant to avoid division by zero
    quot = df.div(ref, axis=1).median(axis=1)
    df = df.div(quot, axis=0)
    print("  - Applied Probabilistic Quotient Normalisation")
    
    # 5. Log2 transform
    df = np.log2(df + 1e-9)  # Add small constant to avoid log(0)
    print("  - Applied log2 transformation")
    
    # 6. Drop any feature that became all-NaN after sample drop
    df = df.dropna(axis=1, how='all')
    
    # 7. Median-impute every remaining NaN
    initial_nans = df.isna().sum().sum()
    df = df.fillna(df.median())
    if initial_nans > 0:
        print(f"  - Imputed {initial_nans} NaN values with median")
    
    # 8. Final check (should pass)
    if df.isna().any().any():
        print("Warning: Still have NaNs after preprocessing!")
        # Force fill any remaining NaNs with 0
        df = df.fillna(0)
    
    print(f"  ✓ Preprocessing complete: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def prepare_ml_data(df: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning
    
    Args:
        df: Feature matrix DataFrame (samples x features)
        labels: Series with sample labels
        
    Returns:
        Tuple of (X, y) arrays ready for ML
    """
    X, y, _ = prepare_ml_data_with_names(df, labels)
    return X, y


def prepare_ml_data_with_names(df: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for machine learning and return feature names
    
    Args:
        df: Feature matrix DataFrame (samples x features)
        labels: Series with sample labels
        
    Returns:
        Tuple of (X, y, feature_names) ready for ML
    """
    # Preprocess the data first
    df_processed = preprocess_dataframe(df)
    
    # Align data and labels (use processed data)
    common_samples = df_processed.index.intersection(labels.index)
    if len(common_samples) == 0:
        raise ModelTrainingError("No common samples between features and labels after preprocessing")
    
    X = df_processed.loc[common_samples].values
    y = labels.loc[common_samples].values
    feature_names = list(df_processed.columns)  # Get feature names AFTER preprocessing
    
    # Convert labels to binary (0/1)
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ModelTrainingError(f"Expected 2 classes, got {len(unique_classes)}: {unique_classes}")
    
    # Convert to binary (Case=1, Control=0 typically)
    y_binary = (y == 'Case').astype(int)
    
    # Check for sufficient samples per class
    unique_classes, counts = np.unique(y_binary, return_counts=True)
    min_class_size = np.min(counts)
    if min_class_size < 5:
        raise ModelTrainingError(f"Minimum class size is {min_class_size}, need at least 5 samples per class")
    
    print(f"✓ Prepared ML data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Binary class distribution: {dict(zip(unique_classes, counts))}")
    
    return X, y_binary, feature_names


def train_logistic_regression(
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Logistic Regression with cross-validation
    
    Args:
        X: Feature matrix
        y: Target labels
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dict with model, scores, and feature importance
    """
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            solver='liblinear',
            random_state=random_state,
            max_iter=1000
        ))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Fit final model
    pipeline.fit(X, y)
    
    # Get feature importance (coefficients)
    coef = pipeline.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature_idx': range(len(coef)),
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Predictions for final ROC
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    final_auc = roc_auc_score(y, y_pred_proba)
    
    return {
        'model_type': 'logistic_regression',
        'pipeline': pipeline,
        'cv_scores': cv_scores,
        'mean_cv_auc': np.mean(cv_scores),
        'std_cv_auc': np.std(cv_scores),
        'final_auc': final_auc,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba
    }


def train_lightgbm(
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train LightGBM with cross-validation
    
    Args:
        X: Feature matrix
        y: Target labels  
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dict with model, scores, and feature importance
    """
    if not LIGHTGBM_AVAILABLE:
        raise ModelTrainingError("LightGBM not available. Install with: pip install lightgbm")
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': random_state
    }
    
    # Cross-validation with LightGBM
    lgb_data = lgb.Dataset(X, label=y)
    cv_results = lgb.cv(
        lgb_params,
        lgb_data,
        num_boost_round=100,
        nfold=cv_folds,
        shuffle=True,
        seed=random_state,
        return_cvbooster=True,
        eval_train_metric=True
    )
    
    # Extract CV scores
    cv_scores = cv_results['valid auc-mean']
    best_iteration = np.argmax(cv_scores)
    best_cv_auc = cv_scores[best_iteration]
    
    # Train final model
    final_model = lgb.train(
        lgb_params,
        lgb_data,
        num_boost_round=best_iteration + 1,
        valid_sets=[lgb_data],
        verbose_eval=False
    )
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature_idx': range(X.shape[1]),
        'importance': final_model.feature_importance(),
    }).sort_values('importance', ascending=False)
    
    # Predictions for final AUC
    y_pred_proba = final_model.predict(X)
    final_auc = roc_auc_score(y, y_pred_proba)
    
    return {
        'model_type': 'lightgbm',
        'model': final_model,
        'cv_scores': cv_scores,
        'best_cv_auc': best_cv_auc,
        'best_iteration': best_iteration,
        'final_auc': final_auc,
        'feature_importance': feature_importance,
        'y_pred_proba': y_pred_proba
    }


def generate_roc_plot(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    output_path: Union[str, Path]
) -> Path:
    """
    Generate ROC curve plot
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_model_artifacts(
    results: Dict[str, Any],
    feature_names: Optional[List[str]],
    output_dir: Union[str, Path],
    model_name: str
) -> Dict[str, Path]:
    """
    Save model artifacts (model, importance, plots)
    
    Args:
        results: Model training results
        feature_names: List of feature names (optional)
        output_dir: Directory to save artifacts
        model_name: Name for the model files
        
    Returns:
        Dict mapping artifact types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_paths = {}
    
    # Save model
    if 'pipeline' in results:
        # Scikit-learn pipeline
        model_path = output_dir / f"{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(results['pipeline'], f)
        artifact_paths['model'] = model_path
        
    elif 'model' in results:
        # LightGBM model
        model_path = output_dir / f"{model_name}_model.txt"
        results['model'].save_model(str(model_path))
        artifact_paths['model'] = model_path
    
    # Save feature importance
    importance_df = results['feature_importance'].copy()
    
    # Add feature names if available
    if feature_names and len(feature_names) == len(importance_df):
        # Map feature indices to actual feature names
        importance_df['feature_name'] = [feature_names[i] for i in importance_df['feature_idx']]
        print(f"  ✓ Added metabolite names to {len(importance_df)} features")
    else:
        # Fallback to generic names
        importance_df['feature_name'] = [f"Feature {i}" for i in importance_df['feature_idx']]
        if feature_names:
            print(f"  ⚠️ Feature name count mismatch: {len(feature_names)} names vs {len(importance_df)} features")
    
    importance_path = output_dir / f"{model_name}_importance.json"
    importance_json = importance_df.to_dict(orient='records')
    with open(importance_path, 'w') as f:
        json.dump(importance_json, f, indent=2)
    artifact_paths['importance'] = importance_path
    
    # Save results summary with feature names
    summary = {
        'model_type': results['model_type'],
        'mean_cv_auc': float(results.get('mean_cv_auc', results.get('best_cv_auc', 0))),
        'final_auc': float(results['final_auc']),
        'n_features': len(importance_df),
        'top_10_features': importance_df.head(10).to_dict(orient='records')
    }
    
    summary_path = output_dir / f"{model_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    artifact_paths['summary'] = summary_path
    
    return artifact_paths


def train_models(
    df: pd.DataFrame,
    labels: pd.Series,
    output_dir: Union[str, Path],
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train both Logistic Regression and LightGBM models
    
    Args:
        df: Feature matrix DataFrame
        labels: Target labels Series
        output_dir: Directory to save results
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dict with results from both models
    """
    # Prepare data (includes preprocessing) and get processed feature names
    X, y, feature_names = prepare_ml_data_with_names(df, labels)
    
    results = {}
    
    # Train Logistic Regression
    print("\n--- Training Logistic Regression ---")
    try:
        lr_results = train_logistic_regression(X, y, cv_folds, random_state)
        print(f"✓ Logistic Regression CV AUC: {lr_results['mean_cv_auc']:.3f} ± {lr_results['std_cv_auc']:.3f}")
        print(f"✓ Final AUC: {lr_results['final_auc']:.3f}")
        
        # Save artifacts
        lr_artifacts = save_model_artifacts(lr_results, feature_names, output_dir, "logistic_regression")
        
        # Generate ROC plot
        roc_path = generate_roc_plot(y, lr_results['y_pred_proba'], 'Logistic Regression', 
                                   Path(output_dir) / "logistic_regression_roc.png")
        lr_artifacts['roc_plot'] = roc_path
        
        results['logistic_regression'] = {
            'results': lr_results,
            'artifacts': lr_artifacts
        }
        
    except Exception as e:
        print(f"Warning: Logistic Regression training failed: {e}")
    
    # Train LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        print("\n--- Training LightGBM ---")
        try:
            lgb_results = train_lightgbm(X, y, cv_folds, random_state)
            print(f"✓ LightGBM CV AUC: {lgb_results['best_cv_auc']:.3f}")
            print(f"✓ Final AUC: {lgb_results['final_auc']:.3f}")
            
            # Save artifacts
            lgb_artifacts = save_model_artifacts(lgb_results, feature_names, output_dir, "lightgbm")
            
            # Generate ROC plot
            roc_path = generate_roc_plot(y, lgb_results['y_pred_proba'], 'LightGBM', 
                                       Path(output_dir) / "lightgbm_roc.png")
            lgb_artifacts['roc_plot'] = roc_path
            
            results['lightgbm'] = {
                'results': lgb_results,
                'artifacts': lgb_artifacts
            }
            
        except Exception as e:
            print(f"Warning: LightGBM training failed: {e}")
    else:
        print("\n--- LightGBM not available ---")
    
    print(f"\n✅ Model training completed! Results saved to: {Path(output_dir).absolute()}")
    return results


# CLI interface
def main():
    """Command line interface for model training"""
    if len(sys.argv) < 2:
        print("Usage: python model_pipeline.py <mwtab_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "model_output"
    
    try:
        # Import required modules
        from biomarker.ingest.file_loader import load_file
        
        # Load data
        print(f"Loading data from {input_file}...")
        df, metadata = load_file(input_file)
        
        # Extract labels (this assumes Case/Control in the data)
        # For now, we'll use a simple heuristic based on sample names or you can modify this
        print("Extracting labels...")
        
        # Try to load labels from the mwTab file
        try:
            import mwtab
            mw = next(mwtab.read_files(input_file))
            ssf = pd.DataFrame(mw['SUBJECT_SAMPLE_FACTORS'])
            factors = pd.json_normalize(ssf['Factors'])
            ssf = ssf.drop(columns='Factors').join(factors)
            labels = ssf.set_index('Sample ID')['Group'].reindex(df.index)
            print(f"✓ Loaded labels: {labels.value_counts().to_dict()}")
        except Exception as e:
            print(f"Warning: Could not extract labels from mwTab: {e}")
            # Create dummy labels for testing (first half = Case, second half = Control)
            n_samples = len(df)
            labels = pd.Series(['Case'] * (n_samples//2) + ['Control'] * (n_samples - n_samples//2), 
                             index=df.index, name='Group')
            print(f"✓ Created dummy labels for testing: {labels.value_counts().to_dict()}")
        
        # Train models
        results = train_models(df, labels, output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL TRAINING SUMMARY")
        print("="*50)
        
        for model_name, model_data in results.items():
            model_results = model_data['results']
            print(f"\n{model_name.upper()}:")
            print(f"  Final AUC: {model_results['final_auc']:.3f}")
            if 'mean_cv_auc' in model_results:
                print(f"  CV AUC: {model_results['mean_cv_auc']:.3f} ± {model_results['std_cv_auc']:.3f}")
            print(f"  Artifacts: {len(model_data['artifacts'])} files")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 