#!/usr/bin/env python3
"""
Task #9: Scaler/Transformer Block
Batch normalization and scaling for ML robustness.

Implements Pareto scaling, Log10 transformation, and other common metabolomics scaling methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings


class ScalingError(Exception):
    """Raised when scaling fails"""
    pass


@dataclass
class ScalerParams:
    """Parameters for different scaling methods"""
    method: str = 'pareto'
    log_transform: bool = False
    log_base: str = 'log10'  # 'log10', 'log2', 'ln'
    log_offset: float = 1e-6  # Small value added before log transform
    center: bool = True
    handle_zeros: str = 'offset'  # 'offset', 'ignore', 'mask'
    clip_outliers: bool = False
    outlier_percentiles: Tuple[float, float] = (1.0, 99.0)


class ParetoScaler(BaseEstimator, TransformerMixin):
    """
    Pareto scaling: divide by square root of standard deviation
    Common in metabolomics to reduce the relative importance of large fold changes
    """
    
    def __init__(self, center: bool = True):
        self.center = center
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit the scaler to the data"""
        X = np.asarray(X)
        
        if self.center:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        # Calculate standard deviation and take square root for Pareto scaling
        std = np.std(X, axis=0, ddof=1)
        self.scale_ = np.sqrt(std)
        
        # Handle zero variance features
        self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using Pareto scaling"""
        if self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet")
        
        X = np.asarray(X)
        
        if self.center:
            X_scaled = (X - self.mean_) / self.scale_
        else:
            X_scaled = X / self.scale_
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the scaled data"""
        if self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet")
        
        X = np.asarray(X)
        
        if self.center:
            return X * self.scale_ + self.mean_
        else:
            return X * self.scale_


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log transformation for metabolomics data
    Handles zeros and negative values appropriately
    """
    
    def __init__(self, base: str = 'log10', offset: float = 1e-6, handle_zeros: str = 'offset'):
        self.base = base
        self.offset = offset
        self.handle_zeros = handle_zeros
        self.min_positive_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit the transformer to determine offset for zeros"""
        X = np.asarray(X)
        
        if self.handle_zeros == 'offset':
            # Find the minimum positive value for offset
            positive_values = X[X > 0]
            if len(positive_values) > 0:
                self.min_positive_ = np.min(positive_values)
            else:
                self.min_positive_ = 1.0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log transformation"""
        X = np.asarray(X).copy()
        
        if self.handle_zeros == 'offset':
            # Add small offset to handle zeros
            offset_value = self.min_positive_ * self.offset if self.min_positive_ else self.offset
            X[X <= 0] = offset_value
            X = X + offset_value
        elif self.handle_zeros == 'mask':
            # Set zeros/negatives to NaN
            X[X <= 0] = np.nan
        # For 'ignore', do nothing - let log handle it
        
        # Apply log transformation
        if self.base == 'log10':
            return np.log10(X)
        elif self.base == 'log2':
            return np.log2(X)
        elif self.base == 'ln':
            return np.log(X)
        else:
            raise ValueError(f"Unsupported log base: {self.base}")
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


def apply_scaling(
    feature_df: pd.DataFrame,
    method: str = 'pareto',
    params: Optional[ScalerParams] = None,
    fit_on: str = 'all'  # 'all', 'train' (future use for train/test split)
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply scaling/transformation to feature matrix
    
    Args:
        feature_df: Feature matrix (samples × features)
        method: Scaling method ('pareto', 'standard', 'minmax', 'robust', 'log10', 'none')
        params: Scaling parameters
        fit_on: What data to fit on ('all' for now)
        
    Returns:
        Tuple of (scaled_df, scaling_info)
    """
    if params is None:
        params = ScalerParams(method=method)
    
    print(f"🔧 Applying {method} scaling...")
    
    # Separate numeric columns for scaling
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = feature_df.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_columns) == 0:
        print("  ⚠️ No numeric columns found - returning original data")
        return feature_df.copy(), {'method': method, 'columns_scaled': 0}
    
    print(f"  Scaling {len(numeric_columns)} numeric features...")
    
    # Extract numeric data
    numeric_data = feature_df[numeric_columns].copy()
    
    # Handle outliers if requested
    if params.clip_outliers:
        numeric_data = clip_outliers(numeric_data, params.outlier_percentiles)
    
    # Apply log transformation first if requested
    log_info = None
    if params.log_transform:
        print(f"  Applying {params.log_base} transformation...")
        log_transformer = LogTransformer(
            base=params.log_base,
            offset=params.log_offset,
            handle_zeros=params.handle_zeros
        )
        numeric_data = pd.DataFrame(
            log_transformer.fit_transform(numeric_data.values),
            index=numeric_data.index,
            columns=numeric_data.columns
        )
        log_info = {
            'base': params.log_base,
            'offset': params.log_offset,
            'handle_zeros': params.handle_zeros,
            'min_positive': log_transformer.min_positive_
        }
    
    # Apply scaling transformation
    scaled_data, scaler_info = apply_scaling_method(numeric_data, method, params)
    
    # Reconstruct full DataFrame
    scaled_df = feature_df.copy()
    scaled_df[numeric_columns] = scaled_data
    
    # Preserve metadata if present
    if 'feature_metadata' in feature_df.attrs:
        scaled_df.attrs['feature_metadata'] = feature_df.attrs['feature_metadata']
    
    # Store scaling information in DataFrame attributes
    scaling_info = {
        'method': method,
        'parameters': params.__dict__,
        'columns_scaled': len(numeric_columns),
        'log_transformation': log_info,
        'scaler_info': scaler_info,
        'original_shape': feature_df.shape,
        'scaled_shape': scaled_df.shape
    }
    
    scaled_df.attrs['scaling_info'] = scaling_info
    
    print(f"  ✅ Scaling complete!")
    print(f"    Method: {method}")
    print(f"    Features scaled: {len(numeric_columns)}")
    if params.log_transform:
        print(f"    Log transform: {params.log_base}")
    
    return scaled_df, scaling_info


def apply_scaling_method(
    data: pd.DataFrame,
    method: str,
    params: ScalerParams
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply specific scaling method to numeric data
    
    Args:
        data: Numeric data to scale
        method: Scaling method
        params: Scaling parameters
        
    Returns:
        Tuple of (scaled_data, scaler_info)
    """
    if method == 'none':
        return data.copy(), {'method': 'none', 'applied': False}
    
    elif method == 'pareto':
        scaler = ParetoScaler(center=params.center)
        
    elif method == 'standard':
        scaler = StandardScaler()
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        
    elif method == 'robust':
        scaler = RobustScaler()
        
    elif method == 'power':
        # Yeo-Johnson power transformation
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        
    else:
        raise ScalingError(f"Unknown scaling method: {method}")
    
    # Fit and transform
    scaled_values = scaler.fit_transform(data.values)
    
    # Create scaled DataFrame
    scaled_data = pd.DataFrame(
        scaled_values,
        index=data.index,
        columns=data.columns
    )
    
    # Extract scaler information
    scaler_info = {
        'method': method,
        'scaler_type': type(scaler).__name__,
        'fitted': True
    }
    
    # Add method-specific information
    if hasattr(scaler, 'mean_'):
        scaler_info['mean'] = scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else float(scaler.mean_)
    if hasattr(scaler, 'scale_'):
        scaler_info['scale'] = scaler.scale_.tolist() if hasattr(scaler.scale_, 'tolist') else float(scaler.scale_)
    if hasattr(scaler, 'center_'):
        scaler_info['center'] = scaler.center_.tolist() if hasattr(scaler.center_, 'tolist') else float(scaler.center_)
    
    return scaled_data, scaler_info


def clip_outliers(
    data: pd.DataFrame,
    percentiles: Tuple[float, float] = (1.0, 99.0)
) -> pd.DataFrame:
    """
    Clip outliers based on percentile thresholds
    
    Args:
        data: Input data
        percentiles: Lower and upper percentile thresholds
        
    Returns:
        Data with outliers clipped
    """
    clipped_data = data.copy()
    
    for column in data.columns:
        lower_bound = np.percentile(data[column].dropna(), percentiles[0])
        upper_bound = np.percentile(data[column].dropna(), percentiles[1])
        
        clipped_data[column] = np.clip(data[column], lower_bound, upper_bound)
    
    return clipped_data


def get_available_scalers() -> Dict[str, str]:
    """
    Get available scaling methods with descriptions
    
    Returns:
        Dictionary of method names and descriptions
    """
    return {
        'none': 'No scaling applied',
        'pareto': 'Pareto scaling (divide by sqrt of std) - common in metabolomics',
        'standard': 'Standard scaling (z-score normalization)',
        'minmax': 'Min-max scaling to [0,1] range',
        'robust': 'Robust scaling using median and IQR',
        'power': 'Yeo-Johnson power transformation'
    }


def compare_scaling_methods(
    feature_df: pd.DataFrame,
    methods: List[str] = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Compare different scaling methods on the same data
    
    Args:
        feature_df: Feature matrix
        methods: List of methods to compare
        sample_size: Number of samples to use for comparison
        
    Returns:
        Comparison results
    """
    if methods is None:
        methods = ['none', 'pareto', 'standard', 'minmax', 'robust']
    
    # Sample data if it's too large
    if len(feature_df) > sample_size:
        sample_df = feature_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = feature_df
    
    results = {}
    
    for method in methods:
        print(f"Comparing {method} scaling...")
        
        try:
            scaled_df, scaling_info = apply_scaling(sample_df, method)
            
            # Calculate comparison metrics
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
            original_data = sample_df[numeric_cols]
            scaled_data = scaled_df[numeric_cols]
            
            metrics = {
                'mean_of_means': float(scaled_data.mean().mean()),
                'mean_of_stds': float(scaled_data.std().mean()),
                'min_value': float(scaled_data.min().min()),
                'max_value': float(scaled_data.max().max()),
                'has_negative': bool((scaled_data < 0).any().any()),
                'has_inf': bool(np.isinf(scaled_data).any().any()),
                'has_nan': bool(scaled_data.isna().any().any()),
                'scaling_info': scaling_info
            }
            
            results[method] = metrics
            
        except Exception as e:
            results[method] = {'error': str(e)}
    
    return results


def save_scaling_report(
    scaling_info: Dict,
    output_file: Path
) -> None:
    """
    Save scaling information to JSON file
    
    Args:
        scaling_info: Scaling information
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(scaling_info, f, indent=2, default=str)
    
    print(f"  📄 Scaling report saved: {output_file}")


# CLI interface
def main():
    """Command line interface for scaling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Scaling")
    parser.add_argument("feature_matrix", help="Input feature matrix CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--method", default='pareto',
                       choices=list(get_available_scalers().keys()),
                       help="Scaling method")
    parser.add_argument("--log-transform", action='store_true', help="Apply log transformation first")
    parser.add_argument("--log-base", default='log10', choices=['log10', 'log2', 'ln'], help="Log base")
    parser.add_argument("--no-center", action='store_true', help="Don't center data before scaling")
    parser.add_argument("--clip-outliers", action='store_true', help="Clip outliers before scaling")
    parser.add_argument("--compare", action='store_true', help="Compare multiple scaling methods")
    
    args = parser.parse_args()
    
    try:
        # Load feature matrix
        print(f"Loading feature matrix from {args.feature_matrix}...")
        feature_df = pd.read_csv(args.feature_matrix, index_col=0)
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.compare:
            # Compare multiple methods
            print("Comparing scaling methods...")
            comparison = compare_scaling_methods(feature_df)
            
            comparison_file = output_dir / "scaling_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            
            print(f"Comparison saved: {comparison_file}")
            
            # Print summary
            for method, results in comparison.items():
                if 'error' in results:
                    print(f"  {method}: ERROR - {results['error']}")
                else:
                    print(f"  {method}: mean={results['mean_of_means']:.3f}, std={results['mean_of_stds']:.3f}")
        
        else:
            # Apply single scaling method
            params = ScalerParams(
                method=args.method,
                log_transform=args.log_transform,
                log_base=args.log_base,
                center=not args.no_center,
                clip_outliers=args.clip_outliers
            )
            
            scaled_df, scaling_info = apply_scaling(feature_df, args.method, params)
            
            # Save results
            output_file = output_dir / "scaled_features.csv"
            scaled_df.to_csv(output_file)
            
            report_file = output_dir / "scaling_report.json"
            save_scaling_report(scaling_info, report_file)
            
            print(f"\n✅ Scaling completed!")
            print(f"Output: {output_file}")
            print(f"Report: {report_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 