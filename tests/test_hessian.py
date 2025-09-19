"""Tests for Hessian computations and utilities."""

import pytest
import torch
from torch import nn
import numpy as np
import random

def test_hessian_imports():
    """Test that the hessian package can be imported if available."""
    try:
        from oxford_loss_landscapes.hessian import (
            min_max_hessian_eigs,
            get_eigenstuff,
            small_hessian,
            eval_hess_vec_prod,
            npvec_to_tensorlist,
            gradtensor_to_npvec,
            get_hessian_eigenstuff,
            create_hessian_vector_product
        )
        from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace

        assert callable(min_max_hessian_eigs)
        assert callable(get_eigenstuff)
        assert callable(small_hessian)
        assert callable(eval_hess_vec_prod)
        assert callable(npvec_to_tensorlist)
        assert callable(gradtensor_to_npvec)
        assert callable(get_hessian_eigenstuff)
        assert callable(create_hessian_vector_product)
        assert callable(hessian_trace)

    except ImportError as e:
        pytest.skip(f"Hessian package not available; skipping related tests. Error: {e}")

def test_eval_hess_vec_prod():
    """Test Hessian computations on a simple model."""
    try:
        from oxford_loss_landscapes.hessian import min_max_hessian_eigs, hessian_trace
    except ImportError as e:
        pytest.skip(f"Hessian package not available; skipping related tests. Error: {e}")

    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
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
    original_params = np.concatenate([p.clone().detach().numpy().flatten() for p in model.parameters()])

    # Store the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())

    new_weights = np.array([random.random() for _ in range(total_params)])

    new_model = copy_wts_into_model(new_weights, model)

    new_model_parameters = np.concatenate([p.clone().detach().numpy().flatten() for p in new_model.parameters()])
    
    # Check that weights have been updated
    for p, new_w in zip(new_model_parameters, new_weights):
        assert np.allclose(p, new_w)
    
    # Check that weights are different from original
    for p, orig_p in zip(new_model_parameters, original_params):
        assert not np.allclose(p, orig_p)

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
    tensor_list = npvec_to_tensorlist(flat_weights, original_weights)
    
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

def test_create_hessian_vector_product_from_loss():
    """Test the new create_hessian_vector_product function."""
    try:
        from oxford_loss_landscapes.hessian.hessian import create_hessian_vector_product_from_loss
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    criterion = nn.MSELoss()
    
    torch.manual_seed(42)
    X = torch.randn(20, 3)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(20, 1)

    loss = criterion(model(X), y)
    
    # Create HVP function
    hvp_func, params, N = create_hessian_vector_product_from_loss(
        model, loss, use_cuda=False, all_params=True
    )
    
    # Test the function
    assert callable(hvp_func)
    assert isinstance(params, list)
    assert isinstance(N, int)
    assert N > 0
    
    # Test HVP computation
    random_vec = np.random.randn(N)
    result = hvp_func(random_vec)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (N,)
    
    # Test linearity: H(2v) should be 2*H(v)
    result2 = hvp_func(2 * random_vec)
    assert np.allclose(result2, 2 * result, rtol=1e-5)


def test_get_eigenstuff_numpy():
    """Test get_eigenstuff function with numpy method."""
    try:
        from oxford_loss_landscapes.hessian.hessian import get_eigenstuff
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a symmetric matrix
    A = np.array([[2, 1], [1, 2]], dtype=np.float32)
    
    eigenvalues, eigenvectors = get_eigenstuff(A, num_eigs_returned=2, method='numpy')
    
    # Check that eigenvalues and eigenvectors are correct
    assert isinstance(eigenvalues, list)
    assert isinstance(eigenvectors, list)
    assert len(eigenvalues) == 2
    assert len(eigenvectors) == 2
    assert np.isclose(eigenvalues[0], 1.0)
    assert np.isclose(eigenvalues[1], 3.0)

def test_get_eigenstuff_scipy():
    """Test get_eigenstuff function with scipy method."""
    try:
        from oxford_loss_landscapes.hessian.hessian import get_eigenstuff
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a symmetric matrix
    A = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float32)
    
    eigenvalues, eigenvectors = get_eigenstuff(A, num_eigs_returned=3, method='scipy')
    
    # Check that eigenvalues and eigenvectors are correct
    assert isinstance(eigenvalues, list)
    assert isinstance(eigenvectors, list)
    assert len(eigenvalues) == 3
    assert len(eigenvectors) == 3
    assert np.isclose(eigenvalues[0], 1.0)
    assert np.isclose(eigenvalues[1], 2.0)
    assert np.isclose(eigenvalues[2], 3.0)

def test_small_hessian():
    """Test small_hessian function on a simple model."""
    try:
        from oxford_loss_landscapes.hessian.hessian import small_hessian
    except ImportError:
        pytest.skip("Hessian package not available; skipping related tests.")
    
    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    n_param = sum(p.numel() for p in model.parameters())

    dummy_input = torch.randn(5, 2)
    dummy_output = model(dummy_input)
    loss = torch.sum(dummy_output)  # This creates a loss that depends on model parameters

    torch.manual_seed(42)

    # Compute all eigenvalues and eigenvectors
    hessian = small_hessian(model, loss=loss)

    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (n_param, n_param)

def test_hessian_vector_product():
    pass

def test_get_eigenstuff():
    try:
        from oxford_loss_landscapes.hessian.hessian import get_eigenstuff
    except ImportError as e:
        pytest.skip(f"Hessian package not available; skipping related tests. Error: {e}")

    A1 = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float32)

    eigval, eigvec = get_eigenstuff(A1, num_eigs_returned=3, method='numpy')

    assert len(eigval) == 3
    assert len(eigvec) == 3
    # Check eigenvalues are approximately correct (they're floats, not integers)
    assert np.allclose(eigval, [1.0, 2.0, 3.0])

    # Use a smaller matrix size for testing (1e4 is too large and slow)
    medium_matrix_size = int(1e2)
    A2 = np.eye(medium_matrix_size)

    # Use numpy method for smaller matrices - scipy has issues with k >= N
    eigval, eigvec = get_eigenstuff(A2, num_eigs_returned=medium_matrix_size, method='numpy')
    assert len(eigval) == medium_matrix_size
    # Check that sum of eigenvalues equals the matrix size (trace of identity matrix)
    assert abs(sum(eigval) - medium_matrix_size) < 1e-6

def test_get_hessian():
    try:
        from oxford_loss_landscapes.hessian.hessian import get_hessian
    except ImportError as e:
        pytest.skip(f"Hessian package not available; skipping related tests. Error: {e}")
    
    # Create a simple model and data
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    model.eval()
    
    # Count total parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Create dummy data
    torch.manual_seed(42)
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    criterion = nn.MSELoss()

    output = model(X)
    loss = criterion(output, y)
    
    # Compute Hessian
    hessian = get_hessian(model,loss, method='numpy')
    
    # Check basic properties
    assert isinstance(hessian, np.ndarray)
    assert hessian.shape == (n_params, n_params)
    
    # Check symmetry (should be symmetric since Hessian is symmetric)
    assert np.allclose(hessian, hessian.T, rtol=1e-5)
    
    # Check if eigenvalues exist
    eigenvalues = np.linalg.eigvals(hessian)
    assert len(eigenvalues) == n_params

    



