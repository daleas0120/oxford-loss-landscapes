#!/usr/bin/env python3
"""
Hessian Usage Demo - Working Example
===================================

This script demonstrates how to use the oxford_loss_landscapes.hessian package
to calculate Hessian eigenvalues and eigenvectors for neural network models.

Note: This example is designed to work with the current environment constraints.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_hessian_imports():
    """Test that all Hessian modules can be imported correctly."""
    print("Testing Hessian Module Imports")
    print("=" * 50)
    
    try:
        from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
        print("✓ Successfully imported min_max_hessian_eigs")
    except ImportError as e:
        print(f"✗ Failed to import min_max_hessian_eigs: {e}")
        return False
    
    try:
        from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
        print("✓ Successfully imported hessian_trace")
    except ImportError as e:
        print(f"✗ Failed to import hessian_trace: {e}")
        return False
    
    try:
        from oxford_loss_landscapes.hessian.utilities import (
            get_weights, set_weights, tensorlist_to_tensor
        )
        print("✓ Successfully imported hessian utilities")
    except ImportError as e:
        print(f"✗ Failed to import hessian utilities: {e}")
        return False
    
    return True

def show_function_signatures():
    """Display the function signatures for key Hessian functions."""
    print("\nHessian Function Signatures")
    print("=" * 50)
    
    from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
    from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
    
    print("1. min_max_hessian_eigs(model_wrapper, loss_fn=None, use_cuda=False, max_iter=100, tol=1e-3)")
    print("   Purpose: Compute the minimum and maximum eigenvalues of the Hessian matrix")
    print("   Returns: (min_eigenvalue, max_eigenvalue, min_eigenvector, max_eigenvector)")
    print()
    
    print("2. hessian_trace(model_wrapper, loss_fn=None, num_random_vectors=100, use_cuda=False)")
    print("   Purpose: Estimate the trace of the Hessian matrix using Hutchinson's method")
    print("   Returns: estimated_trace (float)")
    print()

def demonstrate_usage_pattern():
    """Show the typical usage pattern for Hessian analysis."""
    print("\nTypical Usage Pattern")
    print("=" * 50)
    
    usage_code = '''
# 1. Import required modules
import torch
import torch.nn as nn
from oxford_loss_landscapes.model_interface.model_wrapper import ModelWrapper
from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace

# 2. Create your model
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 3. Prepare data
X = torch.randn(100, 4)
y = torch.randn(100, 1)

# 4. Define loss function
def loss_fn(model_output):
    return nn.MSELoss()(model_output, y)

# 5. Wrap your model
model_wrapper = ModelWrapper(model, loss_fn, X)

# 6. Compute Hessian eigenvalues
try:
    min_eig, max_eig, min_eigvec, max_eigvec = min_max_hessian_eigs(
        model_wrapper, 
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"Minimum eigenvalue: {min_eig}")
    print(f"Maximum eigenvalue: {max_eig}")
    print(f"Condition number: {max_eig / min_eig if min_eig > 0 else 'inf'}")
    
except Exception as e:
    print(f"Error computing eigenvalues: {e}")

# 7. Estimate Hessian trace
try:
    trace = hessian_trace(model_wrapper, num_random_vectors=50)
    print(f"Hessian trace estimate: {trace}")
    
except Exception as e:
    print(f"Error computing trace: {e}")
'''
    
    print("Example code:")
    print(usage_code)

def show_interpretation_guide():
    """Provide guidance on interpreting Hessian analysis results."""
    print("\nInterpreting Hessian Analysis Results")
    print("=" * 50)
    
    interpretation = '''
1. EIGENVALUES:
   - Minimum eigenvalue: Indicates the "flattest" direction in loss landscape
     * Negative: Saddle point or local maximum
     * Near zero: Very flat region, potential optimization challenges
     * Positive: Local minimum (convex behavior)
   
   - Maximum eigenvalue: Indicates the "sharpest" direction
     * Large values suggest steep loss changes in some directions
     * Affects optimization stability and learning rate selection

2. CONDITION NUMBER (max_eig / min_eig):
   - Measures optimization difficulty
   - Higher values indicate more challenging optimization
   - Values > 1000 often suggest numerical instability

3. EIGENVECTORS:
   - Show the directions of extreme curvature
   - Can be used to understand which parameters contribute most
   - Useful for parameter space analysis and optimization

4. TRACE:
   - Sum of all eigenvalues
   - Relates to overall "sharpness" of the loss landscape
   - Useful for comparing different models or training states

5. PRACTICAL APPLICATIONS:
   - Learning rate selection: Smaller for larger max eigenvalue
   - Optimization method choice: Second-order methods for high condition numbers
   - Model comparison: Compare sharpness across architectures
   - Training dynamics: Monitor how landscape changes during training
'''
    
    print(interpretation)

def main():
    """Run the Hessian usage demonstration."""
    print("Oxford Loss Landscapes - Hessian Package Demo")
    print("=" * 50)
    print()
    
    # Test imports
    if not test_hessian_imports():
        print("\n✗ Import tests failed. Please check your installation.")
        return
    
    # Show function signatures
    show_function_signatures()
    
    # Demonstrate usage pattern
    demonstrate_usage_pattern()
    
    # Show interpretation guide
    show_interpretation_guide()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Ensure PyTorch and NumPy compatibility (numpy<2.0)")
    print("2. Try the usage pattern with your own model")
    print("3. Experiment with different hyperparameters")
    print("4. Use results to inform optimization strategy")

if __name__ == "__main__":
    main()
