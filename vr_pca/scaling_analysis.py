#!/usr/bin/env python3
"""
Improved scaling law analysis with runtime vs error parametric plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse
from typing import Tuple, Dict, Any
import warnings
# Suppress only specific warnings known to be harmless (e.g., RuntimeWarning from numpy during curve fitting)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


def power_law_model(x, a, b):
    """Simple power law: y = a * x^b"""
    return a * (x ** b)


def power_law_with_offset(x, a, b, c):
    """Power law with constant offset: y = a * x^b + c"""
    return a * (x ** b) + c


def fit_models(x_data: np.ndarray, y_data: np.ndarray, min_n_for_subset: int = None) -> Dict[str, Any]:
    """Fit multiple models and return the best one"""
    results = {}
    
    # Helper function to fit a single model
    def fit_single_model(model_func, name, initial_guess=None, subset_mask=None):
        try:
            x_fit = x_data[subset_mask] if subset_mask is not None else x_data
            y_fit = y_data[subset_mask] if subset_mask is not None else y_data
            
            if initial_guess is None:
                if model_func == power_law_model:
                    # Better initial guess using log-linear fit
                    log_x = np.log(x_fit)
                    log_y = np.log(y_fit)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    initial_guess = [np.exp(coeffs[1]), coeffs[0]]
                elif model_func == power_law_with_offset:
                    # Start with power law fit, add small offset
                    log_x = np.log(x_fit)
                    log_y = np.log(y_fit)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    initial_guess = [np.exp(coeffs[1]), coeffs[0], y_fit.min() * 0.1]
            
            popt, pcov = curve_fit(model_func, x_fit, y_fit, 
                                  p0=initial_guess, maxfev=10000)
            
            # Evaluate on full range for comparison
            y_pred_full = model_func(x_data, *popt)
            y_pred_fit = model_func(x_fit, *popt)
            
            # Calculate R² on both full data and fitting subset
            r2_full = r2_score(y_data, y_pred_full)
            r2_fit = r2_score(y_fit, y_pred_fit)
            
            # Calculate RMSE in log space for better comparison
            log_rmse = np.sqrt(np.mean((np.log(y_data) - np.log(y_pred_full))**2))
            
            param_errors = np.sqrt(np.diag(pcov)) if pcov.shape[0] == len(popt) else [np.nan] * len(popt)
            
            return {
                'name': name,
                'params': popt,
                'errors': param_errors,
                'r2_full': r2_full,
                'r2_fit': r2_fit,
                'log_rmse': log_rmse,
                'y_pred': y_pred_full,
                'subset_used': subset_mask is not None,
                'n_points': len(y_fit),
                'success': True
            }
            
        except Exception as e:
            return {
                'name': name,
                'success': False,
                'error': str(e)
            }
    
    # Fit 1: Standard power law on all data
    results['power_law_full'] = fit_single_model(power_law_model, "Power Law (full)")
    
    # Fit 2: Power law with offset on all data
    results['power_law_offset_full'] = fit_single_model(power_law_with_offset, "Power Law + Offset (full)")
    
    # Fit 3: Power law on subset (excluding smallest N values)
    if min_n_for_subset is not None and len(x_data) > 5:
        subset_mask = x_data >= min_n_for_subset
        if np.sum(subset_mask) >= 4:  # Need at least 4 points
            results['power_law_subset'] = fit_single_model(
                power_law_model, f"Power Law (N≥{min_n_for_subset})", subset_mask=subset_mask)
            
            results['power_law_offset_subset'] = fit_single_model(
                power_law_with_offset, f"Power Law + Offset (N≥{min_n_for_subset})", subset_mask=subset_mask)
    
    # Choose best model based on R² and log_rmse
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        # Prioritize models with offset, then by R²
        def score_model(result):
            offset_bonus = 0.02 if 'offset' in result['name'].lower() else 0
            subset_penalty = -0.01 if result.get('subset_used', False) else 0
            return result['r2_full'] + offset_bonus + subset_penalty
        
        best_key = max(successful_results.keys(), key=lambda k: score_model(successful_results[k]))
        best_result = successful_results[best_key]
        best_result['is_best'] = True
        
        return best_result, results
    else:
        return None, results


def format_equation(result, method_name):
    """Format the equation string for display"""
    if not result or not result.get('success', False):
        return f"{method_name}: Fit failed"
    
    params = result['params']
    errors = result.get('errors', [])
    r2 = result['r2_full']
    name = result['name']
    
    if len(params) == 2:  # Power law
        a, b = params
        if a >= 1e-3:
            a_str = f"{a:.3f}"
        else:
            a_str = f"{a:.2e}"
        equation = f"{method_name}: t = {a_str} × N^{b:.2f}"
    elif len(params) == 3:  # Power law with offset
        a, b, c = params
        if a >= 1e-3:
            a_str = f"{a:.3f}"
        else:
            a_str = f"{a:.2e}"
        if abs(c) >= 1e-3:
            c_str = f"{c:+.3f}"
        else:
            c_str = f"{c:+.2e}"
        equation = f"{method_name}: t = {a_str} × N^{b:.2f} {c_str}"
    else:
        equation = f"{method_name}: Complex model"
    
    model_info = f"({name})" if 'subset' in name.lower() or 'offset' in name.lower() else ""
    r2_str = f"R² = {r2:.3f}"
    
    return f"{equation} {model_info}\n{r2_str}"


def plot_runtime_vs_error(df: pd.DataFrame, output_file: str = "runtime_vs_error.png"):
    """Create a parametric plot of runtime vs error for each N"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Extract data
    N_values = df['N'].values
    eigsh_times = df['eigsh_time_s'].values
    vrpca_times = df['vrpca_time_s'].values
    vrpca_errors = df['rel_err_vrpca'].values
    
    # Eigsh is the reference, so its error is effectively 0
    eigsh_errors = np.zeros_like(eigsh_times)
    
    # Create color map based on N values
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    
    # Plot eigsh points (error = 0)
    for i, (N, time, error, color) in enumerate(zip(N_values, eigsh_times, eigsh_errors, colors)):
        ax.scatter(time, error, color=color, marker='o', s=100, 
                  edgecolors='black', linewidth=1, alpha=0.8, label='Eigsh' if i == 0 else "")
    
    # Plot VR-PCA points
    for i, (N, time, error, color) in enumerate(zip(N_values, vrpca_times, vrpca_errors, colors)):
        ax.scatter(time, error, color=color, marker='^', s=100, 
                  edgecolors='black', linewidth=1, alpha=0.8, label='VR-PCA' if i == 0 else "")
    
    # Connect corresponding points with lines to show the trade-off
    for i, (N, eigsh_t, vrpca_t, vrpca_err, color) in enumerate(
        zip(N_values, eigsh_times, vrpca_times, vrpca_errors, colors)):
        ax.plot([eigsh_t, vrpca_t], [0, vrpca_err], 
               color=color, alpha=0.6, linewidth=1, linestyle='-')
    
    # Add N value annotations for selected points
    n_annotate = min(len(N_values), 8)  # Don't overcrowd
    indices = np.linspace(0, len(N_values)-1, n_annotate, dtype=int)
    
    for i in indices:
        N = N_values[i]
        eigsh_t = eigsh_times[i]
        vrpca_t = vrpca_times[i]
        vrpca_err = vrpca_errors[i]
        color = colors[i]
        
        # Annotate eigsh point
        ax.annotate(f'N={N}', xy=(eigsh_t, 0), xytext=(5, 10), 
                   textcoords='offset points', fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        # Annotate VR-PCA point
        ax.annotate(f'N={N}', xy=(vrpca_t, vrpca_err), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Formatting
    ax.set_xlabel('Runtime (seconds)', fontsize=14)
    ax.set_ylabel('Relative Error', fontsize=14)
    ax.set_title('Runtime vs Accuracy Trade-off\n(Each color represents a different problem size N)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add colorbar to show N progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=N_values.min(), vmax=N_values.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Hidden Size N', fontsize=12)
    
    # Add some analysis text
    fastest_eigsh = eigsh_times.min()
    fastest_vrpca = vrpca_times.min()
    best_vrpca_accuracy = vrpca_errors.min()
    worst_vrpca_accuracy = vrpca_errors.max()
    
    info_text = f"Runtime range:\n"
    info_text += f"Eigsh: {fastest_eigsh:.3f}s - {eigsh_times.max():.1f}s\n"
    info_text += f"VR-PCA: {fastest_vrpca:.3f}s - {vrpca_times.max():.1f}s\n\n"
    info_text += f"VR-PCA error range:\n{best_vrpca_accuracy:.2e} - {worst_vrpca_accuracy:.2e}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved runtime vs error plot: {output_file}")


def plot_combined_scaling(df: pd.DataFrame, output_file: str = "combined_scaling_fixed.png"):
    """Create a single plot with both methods and their best fits"""
    
    plt.figure(figsize=(12, 8))
    
    x_data = df['N'].values
    eigsh_data = df['eigsh_time_s'].values
    vrpca_data = df['vrpca_time_s'].values
    
    # Determine subset threshold - use around 20th percentile of N values
    n_threshold = np.percentile(x_data, 25)
    
    # Fit models
    print("Fitting Eigsh models...")
    eigsh_best, eigsh_all = fit_models(x_data, eigsh_data, min_n_for_subset=n_threshold)
    
    print("Fitting VR-PCA models...")
    vrpca_best, vrpca_all = fit_models(x_data, vrpca_data, min_n_for_subset=n_threshold)
    
    # Plot data points
    plt.loglog(x_data, eigsh_data, 'o', markersize=8, linewidth=2, 
               label='Eigsh data', color='blue', markerfacecolor='lightblue', 
               markeredgecolor='blue', markeredgewidth=2, alpha=0.8)
    
    plt.loglog(x_data, vrpca_data, '^', markersize=8, linewidth=2, 
               label='VR-PCA data', color='red', markerfacecolor='lightcoral', 
               markeredgecolor='red', markeredgewidth=2, alpha=0.8)
    
    # Plot fit lines
    x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
    
    if eigsh_best and eigsh_best.get('success', False):
        params = eigsh_best['params']
        if len(params) == 2:
            y_fit_eigsh = power_law_model(x_fit, *params)
        elif len(params) == 3:
            y_fit_eigsh = power_law_with_offset(x_fit, *params)
        
        plt.loglog(x_fit, y_fit_eigsh, '--', linewidth=3, color='blue', alpha=0.9,
                   label='Eigsh fit')
    
    if vrpca_best and vrpca_best.get('success', False):
        params = vrpca_best['params']
        if len(params) == 2:
            y_fit_vrpca = power_law_model(x_fit, *params)
        elif len(params) == 3:
            y_fit_vrpca = power_law_with_offset(x_fit, *params)
        
        plt.loglog(x_fit, y_fit_vrpca, '--', linewidth=3, color='red', alpha=0.9,
                   label='VR-PCA fit')
    
    # Formatting
    plt.xlabel('Hidden size N', fontsize=14)
    plt.ylabel('Runtime (seconds)', fontsize=14)
    plt.title('Hessian Eigenvalue Computation: Improved Scaling Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    
    # Add equations as text - positioned better
    text_y_start = 0.97
    text_spacing = 0.18
    
    if eigsh_best:
        eq_text = format_equation(eigsh_best, 'Eigsh')
        plt.text(0.02, text_y_start, eq_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        text_y_start -= text_spacing
    
    if vrpca_best:
        eq_text = format_equation(vrpca_best, 'VR-PCA')
        plt.text(0.02, text_y_start, eq_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved improved scaling plot: {output_file}")
    
    return eigsh_best, eigsh_all, vrpca_best, vrpca_all


def create_combined_figure(df: pd.DataFrame, output_file: str = "combined_analysis.png"):
    """Create a combined figure with both scaling and runtime-error plots"""
    
    fig = plt.figure(figsize=(20, 8))
    
    # Left subplot: Scaling analysis
    ax1 = plt.subplot(1, 2, 1)
    
    x_data = df['N'].values
    eigsh_data = df['eigsh_time_s'].values
    vrpca_data = df['vrpca_time_s'].values
    
    # Determine subset threshold
    n_threshold = np.percentile(x_data, 25)
    
    # Fit models
    eigsh_best, _ = fit_models(x_data, eigsh_data, min_n_for_subset=n_threshold)
    vrpca_best, _ = fit_models(x_data, vrpca_data, min_n_for_subset=n_threshold)
    
    # Plot data points
    ax1.loglog(x_data, eigsh_data, 'o', markersize=8, linewidth=2, 
               label='Eigsh data', color='blue', markerfacecolor='lightblue', 
               markeredgecolor='blue', markeredgewidth=2, alpha=0.8)
    
    ax1.loglog(x_data, vrpca_data, '^', markersize=8, linewidth=2, 
               label='VR-PCA data', color='red', markerfacecolor='lightcoral', 
               markeredgecolor='red', markeredgewidth=2, alpha=0.8)
    
    # Plot fit lines
    x_fit = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
    
    if eigsh_best and eigsh_best.get('success', False):
        params = eigsh_best['params']
        if len(params) == 2:
            y_fit_eigsh = power_law_model(x_fit, *params)
        elif len(params) == 3:
            y_fit_eigsh = power_law_with_offset(x_fit, *params)
        
        ax1.loglog(x_fit, y_fit_eigsh, '--', linewidth=3, color='blue', alpha=0.9)
    
    if vrpca_best and vrpca_best.get('success', False):
        params = vrpca_best['params']
        if len(params) == 2:
            y_fit_vrpca = power_law_model(x_fit, *params)
        elif len(params) == 3:
            y_fit_vrpca = power_law_with_offset(x_fit, *params)
        
        ax1.loglog(x_fit, y_fit_vrpca, '--', linewidth=3, color='red', alpha=0.9)
    
    ax1.set_xlabel('Hidden size N', fontsize=14)
    ax1.set_ylabel('Runtime (seconds)', fontsize=14)
    ax1.set_title('Scaling Analysis', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add equations
    text_y_start = 0.97
    text_spacing = 0.18
    
    if eigsh_best:
        eq_text = format_equation(eigsh_best, 'Eigsh')
        ax1.text(0.02, text_y_start, eq_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9))
        text_y_start -= text_spacing
    
    if vrpca_best:
        eq_text = format_equation(vrpca_best, 'VR-PCA')
        ax1.text(0.02, text_y_start, eq_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9))
    
    # Right subplot: Runtime vs Error
    ax2 = plt.subplot(1, 2, 2)
    
    # Extract data for runtime vs error plot
    N_values = df['N'].values
    eigsh_times = df['eigsh_time_s'].values
    vrpca_times = df['vrpca_time_s'].values
    vrpca_errors = df['rel_err_vrpca'].values
    eigsh_errors = np.zeros_like(eigsh_times)
    
    # Create color map based on N values
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    
    # Plot points
    for i, (N, eigsh_t, vrpca_t, vrpca_err, color) in enumerate(
        zip(N_values, eigsh_times, vrpca_times, vrpca_errors, colors)):
        ax2.scatter(eigsh_t, 0, color=color, marker='o', s=80, 
                   edgecolors='black', linewidth=1, alpha=0.8, label='Eigsh' if i == 0 else "")
        ax2.scatter(vrpca_t, vrpca_err, color=color, marker='^', s=80, 
                   edgecolors='black', linewidth=1, alpha=0.8, label='VR-PCA' if i == 0 else "")
        ax2.plot([eigsh_t, vrpca_t], [0, vrpca_err], 
               color=color, alpha=0.6, linewidth=1, linestyle='-')
    
    # Add selected annotations
    n_annotate = min(len(N_values), 6)
    indices = np.linspace(0, len(N_values)-1, n_annotate, dtype=int)
    
    for i in indices:
        N = N_values[i]
        vrpca_t = vrpca_times[i]
        vrpca_err = vrpca_errors[i]
        color = colors[i]
        
        ax2.annotate(f'N={N}', xy=(vrpca_t, vrpca_err), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Runtime (seconds)', fontsize=14)
    ax2.set_ylabel('Relative Error', fontsize=14)
    ax2.set_title('Runtime vs Accuracy Trade-off', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=N_values.min(), vmax=N_values.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label('Hidden Size N', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined analysis plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced scaling analysis with runtime vs error plot")
    parser.add_argument("csv_file", help="Input CSV file with experimental results")
    parser.add_argument("--scaling-plot", default="scaling_analysis.png", help="Scaling plot filename")
    parser.add_argument("--error-plot", default="runtime_vs_error.png", help="Runtime vs error plot filename")
    parser.add_argument("--combined-plot", default="combined_analysis.png", help="Combined plot filename")
    parser.add_argument("--plot-type", choices=['scaling', 'error', 'combined', 'all'], 
                       default='all', help="Which plots to generate")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} data points")
    print(f"N range: {df['N'].min()} to {df['N'].max()}")
    
    # Check required columns
    required_cols = ['N', 'eigsh_time_s', 'vrpca_time_s', 'rel_err_vrpca']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        return
    
    # Generate requested plots
    if args.plot_type in ['scaling', 'all']:
        print("\nCreating scaling analysis plot...")
        plot_combined_scaling(df, args.scaling_plot)
    
    if args.plot_type in ['error', 'all']:
        print("\nCreating runtime vs error plot...")
        plot_runtime_vs_error(df, args.error_plot)
    
    if args.plot_type in ['combined', 'all']:
        print("\nCreating combined analysis plot...")
        create_combined_figure(df, args.combined_plot)


if __name__ == "__main__":
    main()
