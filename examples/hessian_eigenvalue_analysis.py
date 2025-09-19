#!/usr/bin/env python3
"""
Practical Hessian Eigenvalue Analysis

This script shows how to practically use the oxford_loss_landscapes.hessian 
package to compute Hessian eigenvalues and eigenvectors for a neural network.

Key Functions:
- min_max_hessian_eigs(): Computes largest and smallest eigenvalues + eigenvectors
- hessian_trace(): Estimates the trace (sum of all eigenvalues)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Allow running without installing the package by injecting src/ into sys.path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

def create_model(input_dim=5, hidden_dim=10):
    """Create a simple feedforward network."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

def generate_data(n_samples=100, input_dim=5):
    """Generate synthetic regression data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    # Simple linear relationship with noise
    weights = torch.randn(input_dim)
    y = (X @ weights).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y

def train_model(model, X, y, epochs=50):
    """Train the model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return criterion

def analyze_hessian_eigenvalues(model, X, y, criterion):
    """
    Compute Hessian eigenvalues using the hessian package.
    
    Args:
        model: Trained PyTorch model
        X: Input data
        y: Target data  
        criterion: Loss function
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    print("\n" + "="*50)
    print("HESSIAN EIGENVALUE ANALYSIS")
    print("="*50)
    
    try:
        # Import the hessian functions
        from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
        from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
        
        print("✓ Successfully imported hessian functions")
        
        # Set model to evaluation mode
        model.eval()
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params} parameters")
        
        # Compute current loss
        with torch.no_grad():
            current_outputs = model(X)
            current_loss = criterion(current_outputs, y)
            print(f"Current loss: {current_loss.item():.6f}")
        
        print("\nComputing extreme eigenvalues...")
        print("(This may take a moment for larger models)")
        
        # Compute min and max eigenvalues
        max_eig, min_eig, max_eigvec, min_eigvec, iterations = min_max_hessian_eigs(
            net=model,
            inputs=X, 
            outputs=y,
            criterion=criterion,
            rank=0,                 # For single process
            use_cuda=torch.cuda.is_available(),
            verbose=True,           # Print iteration info
            all_params=True         # Include all parameters
        )
        
        # Analyze results
        print(f"\n✓ Eigenvalue computation completed!")
        print(f"Iterations required: {iterations}")
        print(f"Maximum eigenvalue: {max_eig:.8f}")
        print(f"Minimum eigenvalue: {min_eig:.8f}")
        
        # Compute derived quantities
        condition_number = abs(max_eig / min_eig) if min_eig != 0 else float('inf')
        print(f"Condition number: {condition_number:.2f}")
        
        # Interpret the critical point
        print(f"\nCritical Point Analysis:")
        if max_eig > 0 and min_eig > 0:
            print("→ Local MINIMUM (both eigenvalues positive)")
            print("  The model is at a local optimum!")
        elif max_eig < 0 and min_eig < 0:
            print("→ Local MAXIMUM (both eigenvalues negative)")
            print("  This is unusual for a trained model.")
        elif max_eig > 0 and min_eig < 0:
            print("→ SADDLE POINT (mixed eigenvalue signs)")
            print("  The loss surface curves up in some directions, down in others.")
        else:
            print("→ Degenerate case (one eigenvalue is zero)")
        
        # Analyze eigenvectors
        print(f"\nEigenvector Analysis:")
        print(f"Max eigenvector shape: {max_eigvec.shape}")
        print(f"Min eigenvector shape: {min_eigvec.shape}")
        print(f"Max eigenvector L2 norm: {np.linalg.norm(max_eigvec):.6f}")
        print(f"Min eigenvector L2 norm: {np.linalg.norm(min_eigvec):.6f}")
        
        # The eigenvectors tell us about parameter sensitivity
        print(f"\nParameter Sensitivity:")
        max_eigvec_flat = max_eigvec.flatten()
        min_eigvec_flat = min_eigvec.flatten()
        
        print(f"Max eigenvector range: [{np.min(max_eigvec_flat):.4f}, {np.max(max_eigvec_flat):.4f}]")
        print(f"Min eigenvector range: [{np.min(min_eigvec_flat):.4f}, {np.max(min_eigvec_flat):.4f}]")
        
        # Estimate Hessian trace
        print(f"\nEstimating Hessian trace...")
        try:
            trace_estimate = hessian_trace(model, criterion, X, y, num_random_vectors=20)
            avg_eigenvalue = trace_estimate / total_params
            print(f"✓ Estimated trace: {trace_estimate:.6f}")
            print(f"Average eigenvalue: {avg_eigenvalue:.6f}")
            
            # Compare with computed eigenvalues
            print(f"\nComparison:")
            print(f"Computed max eigenvalue: {max_eig:.6f}")
            print(f"Computed min eigenvalue: {min_eig:.6f}")
            print(f"Estimated avg eigenvalue: {avg_eigenvalue:.6f}")
            
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"✗ Error computing trace: {e}")
            trace_estimate = None

        try:
            from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig, top_hessian_eigenpair_vrpca

            print(f"\nRunning VR-PCA solver...")
            vrpca_result = top_hessian_eigenpair_vrpca(
                net=model,
                inputs=X,
                targets=y,
                criterion=criterion,
                config=VRPCAConfig(batch_size=X.shape[0], epochs=8),
            )
            print(
                "VR-PCA dominant eigenvalue: {:.6f} | converged={} | hvp_equiv={:.2f}".format(
                    vrpca_result.eigenvalue,
                    vrpca_result.converged,
                    vrpca_result.hvp_equivalent_calls,
                )
            )
        except ImportError:
            print("VR-PCA solver not available; install the package in editable mode to enable it.")

        return {
            'max_eigenvalue': max_eig,
            'min_eigenvalue': min_eig,
            'max_eigenvector': max_eigvec,
            'min_eigenvector': min_eigvec,
            'condition_number': condition_number,
            'iterations': iterations,
            'trace_estimate': trace_estimate,
            'current_loss': current_loss.item()
        }
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure the oxford_loss_landscapes package is properly installed.")
        return None
        
    except (ValueError, RuntimeError, TypeError, torch.cuda.CudaError, np.linalg.LinAlgError) as e:
        print(f"✗ Computation error: {e}")
        print("This might be due to numerical issues or incompatible NumPy versions.")
        return None

def main():
    """Main function to run the Hessian eigenvalue analysis example."""
    
    print("Practical Hessian Eigenvalue Analysis")
    print("="*50)
    
    # Create model and data
    print("1. Setting up model and data...")
    model = create_model(input_dim=4, hidden_dim=8)
    X, y = generate_data(n_samples=50, input_dim=4)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Train the model
    print("\n2. Training model...")
    criterion = train_model(model, X, y, epochs=30)
    
    # Analyze Hessian
    print("\n3. Analyzing Hessian eigenvalues...")
    results = analyze_hessian_eigenvalues(model, X, y, criterion)
    
    if results:
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Final loss: {results['current_loss']:.6f}")
        print(f"Max eigenvalue: {results['max_eigenvalue']:.6f}")
        print(f"Min eigenvalue: {results['min_eigenvalue']:.6f}")
        print(f"Condition number: {results['condition_number']:.2f}")
        if results['trace_estimate']:
            print(f"Trace estimate: {results['trace_estimate']:.6f}")
        print(f"Convergence iterations: {results['iterations']}")
        
        print(f"\nKey Insights:")
        print(f"• The model converged to a {'local minimum' if results['max_eigenvalue'] > 0 and results['min_eigenvalue'] > 0 else 'saddle point'}")
        print(f"• Condition number of {results['condition_number']:.1f} indicates {'good' if results['condition_number'] < 100 else 'poor'} conditioning")
        print(f"• Eigenvalue magnitudes suggest {'moderate' if abs(results['max_eigenvalue']) < 100 else 'high'} curvature")
        
        print("\n✓ Hessian analysis completed successfully!")
    else:
        print("\n✗ Hessian analysis failed")
        print("\nTroubleshooting tips:")
        print("1. Ensure NumPy compatibility (try downgrading to numpy<2.0)")
        print("2. Check that scipy is installed for eigenvalue computation")
        print("3. Try with a smaller model if memory issues occur")
        print("4. Verify the oxford_loss_landscapes package installation")

if __name__ == "__main__":
    main()
