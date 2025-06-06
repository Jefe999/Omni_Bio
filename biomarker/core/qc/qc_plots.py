#!/usr/bin/env python3
"""
QC Plot Generation for OmniBio MVP
Generates TIC (Total Ion Current) and BPC (Base Peak Chromatogram) plots
For mwTab data, creates synthetic chromatogram-like visualizations
"""

import os
import sys
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings


class QCPlotError(Exception):
    """Raised when QC plot generation fails"""
    pass


def simulate_tic_bpc_from_features(df: pd.DataFrame, sample_name: str) -> Dict[str, np.ndarray]:
    """
    Simulate TIC/BPC traces from feature matrix (for mwTab data)
    This creates synthetic chromatogram-like plots for demo purposes
    
    Args:
        df: Feature matrix DataFrame
        sample_name: Name of the sample
        
    Returns:
        Dict with synthetic TIC/BPC data
    """
    # Create synthetic retention time axis (0-30 minutes)
    n_points = 300
    rt_times = np.linspace(0, 30, n_points)
    
    if sample_name not in df.index:
        raise QCPlotError(f"Sample {sample_name} not found in data")
    
    sample_data = df.loc[sample_name].dropna()
    
    # Simulate TIC by creating peaks at random retention times
    np.random.seed(hash(sample_name) % 2**32)  # Reproducible based on sample name
    
    tic_intensities = np.zeros(n_points)
    bpc_intensities = np.zeros(n_points)
    
    # Create several peaks across the chromatogram
    n_peaks = min(20, len(sample_data) // 10)  # Reasonable number of peaks
    peak_positions = np.random.uniform(0, 30, n_peaks)
    peak_widths = np.random.uniform(0.5, 2.0, n_peaks)
    
    for i, (pos, width) in enumerate(zip(peak_positions, peak_widths)):
        # Use actual feature intensities scaled appropriately
        if i < len(sample_data):
            base_intensity = abs(float(sample_data.iloc[i % len(sample_data)]))
        else:
            base_intensity = np.random.uniform(1000, 100000)
        
        # Create Gaussian peak
        peak = base_intensity * np.exp(-0.5 * ((rt_times - pos) / width) ** 2)
        tic_intensities += peak
        
        # BPC tracks the maximum at each time point
        bpc_intensities = np.maximum(bpc_intensities, peak)
    
    # Add some noise
    noise_level = np.max(tic_intensities) * 0.05
    tic_intensities += np.random.normal(0, noise_level, n_points)
    bpc_intensities += np.random.normal(0, noise_level * 0.3, n_points)
    
    # Ensure no negative values
    tic_intensities = np.maximum(tic_intensities, 0)
    bpc_intensities = np.maximum(bpc_intensities, 0)
    
    return {
        'tic_times': rt_times,
        'tic_intensities': tic_intensities,
        'bpc_times': rt_times,
        'bpc_intensities': bpc_intensities
    }


def generate_single_tic_bpc_plot(
    traces: Dict[str, np.ndarray], 
    sample_name: str, 
    output_dir: Path
) -> Path:
    """
    Generate individual TIC/BPC plot for a single sample
    
    Args:
        traces: Dict with TIC/BPC trace data
        sample_name: Name of the sample
        output_dir: Output directory
        
    Returns:
        Path to generated plot file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'QC Plots - {sample_name}', fontsize=14, fontweight='bold')
    
    # TIC plot
    ax1.plot(traces['tic_times'], traces['tic_intensities'], 'b-', linewidth=1)
    ax1.set_title('Total Ion Current (TIC)', fontweight='bold')
    ax1.set_xlabel('Retention Time (min)')
    ax1.set_ylabel('Intensity')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # BPC plot
    ax2.plot(traces['bpc_times'], traces['bpc_intensities'], 'r-', linewidth=1)
    ax2.set_title('Base Peak Chromatogram (BPC)', fontweight='bold')
    ax2.set_xlabel('Retention Time (min)')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{sample_name}_tic_bpc.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def generate_tic_bpc_plots(
    df: pd.DataFrame, 
    output_dir: Union[str, Path],
    sample_names: Optional[List[str]] = None,
    max_samples: int = 5
) -> List[Path]:
    """
    Generate TIC and BPC plots for samples from mwTab data
    
    Args:
        df: Feature matrix DataFrame (samples x features)
        output_dir: Directory to save plots
        sample_names: List of sample names (if None, uses first few samples)
        max_samples: Maximum number of samples to process
        
    Returns:
        List of generated plot file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Determine which samples to process
    if sample_names is None:
        samples_to_process = df.index.tolist()[:max_samples]
    else:
        samples_to_process = sample_names[:max_samples]
    
    print(f"Generating QC plots for {len(samples_to_process)} samples...")
    
    for sample_name in samples_to_process:
        try:
            print(f"  Processing sample: {sample_name}")
            traces = simulate_tic_bpc_from_features(df, sample_name)
            plot_file = generate_single_tic_bpc_plot(traces, sample_name, output_dir)
            generated_files.append(plot_file)
            
        except Exception as e:
            print(f"Warning: Failed to generate plot for {sample_name}: {e}")
            continue
    
    print(f"✓ Generated {len(generated_files)} QC plots in {output_dir}")
    return generated_files


def generate_qc_summary_plot(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    max_samples: int = 10
) -> Path:
    """
    Generate a summary QC plot showing multiple samples overlaid
    
    Args:
        df: Feature matrix DataFrame
        output_dir: Directory to save plots
        max_samples: Maximum number of samples to include
        
    Returns:
        Path to summary plot file
    """
    output_dir = Path(output_dir)
    samples_to_plot = df.index.tolist()[:max_samples]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('QC Summary - All Samples Overlay', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(samples_to_plot)))
    
    for i, sample_name in enumerate(samples_to_plot):
        try:
            traces = simulate_tic_bpc_from_features(df, sample_name)
            
            # TIC overlay
            ax1.plot(traces['tic_times'], traces['tic_intensities'], 
                    color=colors[i], alpha=0.7, linewidth=1, label=sample_name)
            
            # BPC overlay
            ax2.plot(traces['bpc_times'], traces['bpc_intensities'], 
                    color=colors[i], alpha=0.7, linewidth=1, label=sample_name)
                    
        except Exception as e:
            print(f"Warning: Skipping {sample_name} in summary: {e}")
            continue
    
    # TIC plot setup
    ax1.set_title('Total Ion Current (TIC) - All Samples', fontweight='bold')
    ax1.set_xlabel('Retention Time (min)')
    ax1.set_ylabel('Intensity (counts per second)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # BPC plot setup
    ax2.set_title('Base Peak Chromatogram (BPC) - All Samples', fontweight='bold')
    ax2.set_xlabel('Retention Time (min)')
    ax2.set_ylabel('Peak Intensity (counts per second)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save summary plot
    summary_file = output_dir / "qc_summary_overlay.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return summary_file


# CLI interface
def main():
    """Command line interface for QC plot generation"""
    if len(sys.argv) < 2:
        print("Usage: python qc_plots.py <mwtab_file> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "qc_output"
    
    try:
        # Import our file loader
        from biomarker.ingest.file_loader import load_file
        
        # Load data
        print(f"Loading data from {input_file}...")
        df, metadata = load_file(input_file)
        
        # Generate QC plots
        print(f"Generating QC plots...")
        plot_files = generate_tic_bpc_plots(df, output_dir, max_samples=5)
        summary_file = generate_qc_summary_plot(df, output_dir, max_samples=5)
        
        print(f"\n✅ QC plots generated successfully!")
        print(f"Individual plots: {len(plot_files)} files")
        print(f"Summary plot: {summary_file}")
        print(f"Output directory: {Path(output_dir).absolute()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 