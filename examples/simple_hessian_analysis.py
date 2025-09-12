#!/usr/bin/env python3
"""
Simple Hessian Eigenvalue Analysis Example

This script demonstrates how to compute the eigenvalues and eigenvectors of the 
Hessian matrix for a neural network model using direct imports from the hessian package.
"""

import torch
import torch.nn as nn
import numpy as np

def create_simple_model():
    """Create a very simple neural network."""
    return nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(), 
        nn.Linear(10, 1)
    )

def generate_simple_data():
    """Generate simple synthetic data."""
    torch.manual_seed(42)
    X = torch.randn(50, 5)
    y = torch.sum(X[:, :2], dim=1, keepdim=True) + 0.1 * torch.randn(50, 1)
    return X, y

def main():
    print("Simple Hessian Eigenvalue Analysis")
    print("=" * 40)
    
    # Create data and model
    X, y = generate_simple_data()
    model = create_simple_model()
    criterion = nn.MSELoss()
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Quick training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Now compute Hessian eigenvalues
    print("\nComputing Hessian eigenvalues...")
    
    try:
        # Import directly from the hessian module
        from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
        from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
        
        # Compute eigenvalues
        max_eig, min_eig, max_eigvec, min_eigvec, iterations = min_max_hessian_eigs(
            model, X, y, criterion, 
            rank=0, 
            use_cuda=False, 
            verbose=True, 
            all_params=True
        )
        
        print(f"\n✓ Results:")
        print(f"Maximum eigenvalue: {max_eig:.6f}")
        print(f"Minimum eigenvalue: {min_eig:.6f}")
        print(f"Condition number: {abs(max_eig / min_eig):.2f}")
        print(f"Iterations: {iterations}")
        
        # Interpret the results
        if max_eig > 0 and min_eig > 0:
            print("→ Local minimum (positive definite Hessian)")
        elif max_eig > 0 and min_eig < 0:
            print("→ Saddle point (indefinite Hessian)")
        else:
            print("→ Other critical point")
        
        # Eigenvector properties
        print(f"\nEigenvector properties:")
        print(f"Max eigenvector shape: {max_eigvec.shape}")
        print(f"Min eigenvector shape: {min_eigvec.shape}")
        print(f"Max eigenvector norm: {np.linalg.norm(max_eigvec):.6f}")
        print(f"Min eigenvector norm: {np.linalg.norm(min_eigvec):.6f}")
        
        # Estimate trace
        print(f"\nComputing Hessian trace...")
        trace = hessian_trace(model, criterion, X, y, num_samples=10)
        print(f"Estimated trace: {trace:.6f}")
        
        print("\n✓ Hessian analysis completed successfully!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Computation error: {e}")

if __name__ == "__main__":
    main()
