import pytest
import torch
import torch.nn as nn
import numpy as np

def test_hessian_imports():
    """Test that the hessian package can be imported if available."""
    try:
        from oxford_loss_landscapes.hessian import (
            min_max_hessian_eigs,
            hessian_trace,
        )
        assert callable(min_max_hessian_eigs)
        assert callable(hessian_trace)
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")

def test_eval_hess_vec_prod():
    """Test Hessian computations on a simple model."""
    try:
        from oxford_loss_landscapes.hessian import min_max_hessian_eigs, hessian_trace
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    tmp_list = []
    for p in model.parameters():
        tmp_list.extend(p.detach().cpu().numpy().flatten())
    model_param_array = np.array(tmp_list)
    criterion = nn.MSELoss()
    
    torch.manual_seed(42)
    X = torch.randn(30, 3)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(30, 1)
    
    # Compute Hessian properties
    max_eig, min_eig, maxeigvec, mineigvec, _ = min_max_hessian_eigs(model, X, y, criterion)
    
    # Basic assertions - allow numpy scalars and Python floats
    assert isinstance(max_eig, (float, np.floating))
    assert isinstance(min_eig, (float, np.floating))
    assert max_eig >= min_eig

    # Flatten eigenvectors for shape comparison
    maxeigvec_flat = maxeigvec.flatten()
    mineigvec_flat = mineigvec.flatten()
    
    assert model_param_array.shape == maxeigvec_flat.shape
    assert model_param_array.shape == mineigvec_flat.shape

def test_hessian_trace():
    """Test Hessian trace computation on a simple model."""
    try:
        from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    criterion = nn.MSELoss()    
    torch.manual_seed(42)
    X = torch.randn(30, 3)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(30, 1)
    # Compute Hessian trace
    estimated_trace = hessian_trace(model, criterion, X, y, num_random_vectors=10)

    # Basic assertions
    assert isinstance(estimated_trace, float)

def test_copy_wts_into_model():
    """Test copying weights into a model."""
    try:
        from oxford_loss_landscapes.hessian.utilities import copy_wts_into_model
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    original_params = [p.clone() for p in model.parameters()]
    
    # Create new weights with the same shape
    new_weights = []
    for p in model.parameters():
        new_weights.append(torch.randn_like(p))
    

    new_model = copy_wts_into_model(new_weights, model)
    
    # Check that weights have been updated
    for p, new_w in zip(new_model.parameters(), new_weights):
        assert torch.allclose(p, new_w)
    
    # Check that weights are different from original
    for p, orig_p in zip(new_model.parameters(), original_params):
        assert not torch.allclose(p, orig_p)