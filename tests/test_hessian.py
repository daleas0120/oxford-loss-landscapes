"""Tests for Hessian computations and utilities."""

import pytest
import torch
from torch import nn
import numpy as np

from oxford_loss_landscapes.hessian import (
    CLASSICAL_AVAILABLE,
    hessian_trace,
    min_max_hessian_eigs,
)


def test_hessian_imports():
    """Test that the hessian package exposes expected entry points."""
    assert callable(hessian_trace)
    if not CLASSICAL_AVAILABLE:
        pytest.skip("Classical eigensolver unavailable; min_max_hessian_eigs raises ImportError.")
    assert callable(min_max_hessian_eigs)


@pytest.mark.skipif(not CLASSICAL_AVAILABLE, reason="Classical Hessian solver requires SciPy")
def test_eval_hess_vec_prod():
    """Test Hessian computations on a simple model."""
    
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

def test_weights_init():
    """Test weights initialization function."""
    try:
        from oxford_loss_landscapes.hessian.utilities import weights_init
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    original_params = [p.clone() for p in model.parameters()]   

    param_idx = 0
    for layer in model:
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights_init(layer)
            # Check that weights have changed
            assert not torch.allclose(layer.weight, original_params[param_idx])
            param_idx += 1
            if layer.bias is not None:
                assert not torch.allclose(layer.bias, original_params[param_idx])
                param_idx += 1
    assert param_idx == len(original_params), "Not all parameters were checked."

def test_get_weights():
    """Test getting weights from a model."""
    try:
        from oxford_loss_landscapes.hessian.utilities import get_weights
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    weights = get_weights(model)
    # Check that weights is a list of tensors
    assert isinstance(weights, list)
    for w in weights:
        assert isinstance(w, torch.Tensor)

def test_set_weights():
    """Test setting weights into a model."""
    try:
        from oxford_loss_landscapes.hessian.utilities import get_weights, set_weights
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    original_weights = get_weights(model)
    
    # Create new weights with the same shape
    new_weights = []
    for w in original_weights:
        new_weights.append(torch.randn_like(w))
    
    set_weights(model, new_weights)
    
    # Check that weights have been updated
    updated_weights = get_weights(model)
    for w, new_w in zip(updated_weights, new_weights):
        assert torch.allclose(w, new_w)
    
    # Check that weights are different from original
    for w, orig_w in zip(updated_weights, original_weights):
        assert not torch.allclose(w, orig_w)

def test_tensorlist_to_tensor():
    """Test concatenation of a list of tensors into a single tensor."""
    try:
        from oxford_loss_landscapes.hessian.utilities import tensorlist_to_tensor
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    weights = []
    total_params = 0
    for p in model.parameters():
        weights.append(p.data)
        total_params += p.numel()
    
    concatenated = tensorlist_to_tensor(weights)
    
    # Check that concatenated is a 1D tensor with correct length
    assert isinstance(concatenated, torch.Tensor)
    assert concatenated.dim() == 1
    assert concatenated.numel() == total_params

def test_npvec_to_tensorlist():
    """Test conversion from numpy vector to tensor list."""
    try:
        from oxford_loss_landscapes.hessian.utilities import npvec_to_tensorlist, get_weights
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    original_weights = get_weights(model)
    
    # Create a flattened numpy array of weights
    flat_weights = np.concatenate([w.cpu().numpy().flatten() for w in original_weights])
    
    # Convert back to tensor list
    tensor_list = npvec_to_tensorlist(torch.tensor(flat_weights, dtype=torch.float32), original_weights)
    
    # Check that tensor_list has the same shapes as original_weights
    assert isinstance(tensor_list, list)
    assert len(tensor_list) == len(original_weights)
    for t, o in zip(tensor_list, original_weights):
        assert t.shape == o.shape
        assert torch.allclose(t, o)

def test_covariance():
    """Test covariance computation between two sets of vectors."""
    try:
        from oxford_loss_landscapes.hessian.utilities import covariance
    except ImportError:
        pytest.skip("Utils package not available; skipping related tests.")
    
    # Create two sets of vectors
    x = torch.ones(10, 10)
    y = torch.ones(10, 10)
    
    cov_matrix = covariance(x, y, number_ensembles=10)
    
    # Check that cov_matrix is a 10x10 tensor
    assert isinstance(cov_matrix, torch.Tensor)
    assert cov_matrix.shape == (10, 10)
    assert torch.allclose(cov_matrix, torch.zeros(10, 10), atol=1e-6)

