"""
Example demonstrating how to use the refactored Hessian-vector product function.
"""

import torch
import torch.nn as nn
import numpy as np
from oxford_loss_landscapes.hessian import create_hessian_vector_product, min_max_hessian_eigs

def example_usage():
    """Demonstrate the refactored Hessian functionality."""
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Generate sample data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    criterion = nn.MSELoss()
    
    print("=== Example: Using Hessian-Vector Product Function ===")
    
    # Method 1: Create and use Hessian-vector product function directly
    print("\n1. Creating Hessian-vector product function...")
    hvp_func, params, N = create_hessian_vector_product(
        model, inputs, targets, criterion, use_cuda=False, all_params=True
    )
    
    print(f"   Model has {N} parameters")
    print(f"   Created HVP function for {len(params)} parameter tensors")
    
    # Use the HVP function
    print("\n2. Computing Hessian-vector products...")
    np.random.seed(42)  # Set seed for reproducibility
    random_vector = np.random.randn(N)
    
    # Compute H * v
    hvp_result1 = hvp_func(random_vector, verbose=True)
    hvp_result2 = hvp_func(random_vector * 2, verbose=True)
    
    print(f"   HVP result 1 shape: {hvp_result1.shape}")
    print(f"   HVP result 2 shape: {hvp_result2.shape}")
    print(f"   Linearity check (should be ~2.0): {np.linalg.norm(hvp_result2) / np.linalg.norm(hvp_result1):.3f}")
    
    # Method 2: Use the original eigenvalue computation (which now uses the refactored code)
    print("\n3. Computing eigenvalues using refactored function...")
    maxeig, mineig, maxeigvec, mineigvec, iterations = min_max_hessian_eigs(
        model, inputs, targets, criterion, verbose=True, all_params=True
    )
    
    print(f"   Max eigenvalue: {maxeig:.6f}")
    print(f"   Min eigenvalue: {mineig:.6f}")
    if mineig != 0:
        print(f"   Condition number: {maxeig/abs(mineig):.3f}")
    else:
        print("   Condition number: undefined (min eigenvalue is zero)")
    print(f"   Iterations used: {iterations}")
    
    print("\n=== Benefits of Refactoring ===")
    print("✅ HVP function can be reused for multiple computations")
    print("✅ More modular design - separate concerns")
    print("✅ Easier to test individual components")
    print("✅ Can be used for custom eigenvalue algorithms")
    print("✅ Supports different verbosity levels per call")

if __name__ == "__main__":
    example_usage()