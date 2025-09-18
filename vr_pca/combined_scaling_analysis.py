#!/usr/bin/env python3
"""
Improved scaling law analysis with better fitting for small-N behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


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
    print(f"\nSaved improved scaling plot: {output_file}")
    #plt.show()
    
    return eigsh_best, eigsh_all, vrpca_best, vrpca_all


def print_detailed_analysis(eigsh_best, eigsh_all, vrpca_best, vrpca_all):
    """Print detailed analysis of all fitted models"""
    print("\n" + "="*80)
    print("DETAILED FITTING ANALYSIS")
    print("="*80)
    
    print("\nEigsh Models:")
    print("-" * 40)
    for name, result in eigsh_all.items():
        if result.get('success', False):
            r2 = result['r2_full']
            log_rmse = result.get('log_rmse', 'N/A')
            n_points = result.get('n_points', len(result.get('params', [])))
            best_marker = " ← BEST" if result.get('is_best', False) else ""
            print(f"{result['name']}: R²={r2:.3f}, log_RMSE={log_rmse:.3f}, n={n_points}{best_marker}")
    
    print("\nVR-PCA Models:")
    print("-" * 40)
    for name, result in vrpca_all.items():
        if result.get('success', False):
            r2 = result['r2_full']
            log_rmse = result.get('log_rmse', 'N/A')
            n_points = result.get('n_points', len(result.get('params', [])))
            best_marker = " ← BEST" if result.get('is_best', False) else ""
            print(f"{result['name']}: R²={r2:.3f}, log_RMSE={log_rmse:.3f}, n={n_points}{best_marker}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if eigsh_best and eigsh_best.get('success', False):
        params = eigsh_best['params']
        if len(params) >= 2:
            exponent = params[1]
            print(f"\nEigsh scaling: N^{exponent:.2f}")
            if exponent > 2.5:
                print("  → Faster than expected O(N²) growth, possible overhead scaling")
            elif 1.5 < exponent <= 2.5:
                print("  → Roughly quadratic scaling, consistent with dense matrix operations")
            else:
                print("  → Sub-quadratic scaling, better than expected")
    
    if vrpca_best and vrpca_best.get('success', False):
        params = vrpca_best['params']
        if len(params) >= 2:
            exponent = params[1]
            print(f"\nVR-PCA scaling: N^{exponent:.2f}")
            if exponent > 2.5:
                print("  → Faster than expected growth, possible implementation overhead")
            elif 1.5 < exponent <= 2.5:
                print("  → Quadratic-like scaling")
            elif 1.0 < exponent <= 1.5:
                print("  → Near-linear scaling, good for stochastic method")
            else:
                print("  → Sub-linear scaling, excellent!")


def main():
    parser = argparse.ArgumentParser(description="Improved scaling analysis with better small-N handling")
    parser.add_argument("csv_file", help="Input CSV file with experimental results")
    parser.add_argument("--output", "-o", default="combined_scaling_fixed.png", help="Output plot filename")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} data points")
    print(f"N range: {df['N'].min()} to {df['N'].max()}")
    
    # Create the improved plot
    eigsh_best, eigsh_all, vrpca_best, vrpca_all = plot_combined_scaling(df, args.output)
    
    # Detailed analysis
    if args.verbose:
        print_detailed_analysis(eigsh_best, eigsh_all, vrpca_best, vrpca_all)


if __name__ == "__main__":
    main()
