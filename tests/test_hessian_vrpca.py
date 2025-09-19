"""Regression tests for VR-PCA Hessian routines."""

from __future__ import annotations

import pytest
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import scipy  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for the baseline comparison")
def test_vrpca_matches_eigsh_on_small_model():
    from oxford_loss_landscapes.hessian import min_max_hessian_eigs
    from oxford_loss_landscapes.hessian.vrpca import (
        VRPCAConfig,
        min_hessian_eigenpair_vrpca,
        top_hessian_eigenpair_vrpca,
    )

    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 1))
    criterion = nn.MSELoss()

    inputs = torch.randn(40, 4)
    targets = torch.randn(40, 1)

    baseline_max, baseline_min, *_ = min_max_hessian_eigs(
        net=model,
        inputs=inputs,
        outputs=targets,
        criterion=criterion,
        rank=0,
        use_cuda=False,
        verbose=False,
        all_params=True,
    )

    config = VRPCAConfig(batch_size=inputs.shape[0], epochs=8, tol=1e-4)
    result = top_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )

    assert result.converged
    assert result.eigenvector.norm().item() == pytest.approx(1.0, rel=1e-3)
    assert result.eigenvalue == pytest.approx(baseline_max, rel=1e-1)
    assert result.hvp_equivalent_calls > 0

    min_result = min_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )

    assert min_result.eigenvalue == pytest.approx(baseline_min, rel=1e-1)
    assert min_result.eigenvector.norm().item() == pytest.approx(1.0, rel=1e-3)
    assert min_result.hvp_equivalent_calls > 0

    dropin_max, dropin_min, *_ = min_max_hessian_eigs(
        net=model,
        inputs=inputs,
        outputs=targets,
        criterion=criterion,
        backend="vrpca",
        vrpca_config=config,
        compute_min=True,
    )

    assert dropin_max == pytest.approx(baseline_max, rel=1e-1)
    assert dropin_min == pytest.approx(baseline_min, rel=1e-1)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for the baseline comparison")
def test_min_max_vrpca_wrapper_structure():
    from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig, min_max_hessian_eigs_vrpca

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))
    criterion = nn.MSELoss()
    inputs = torch.randn(20, 3)
    targets = torch.randn(20, 1)

    config = VRPCAConfig(batch_size=inputs.shape[0], epochs=5)
    max_eig, min_eig, max_vec, min_vec, iterations = min_max_hessian_eigs_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
        compute_min=True,
    )

    assert isinstance(max_eig, float)
    assert isinstance(min_eig, float)
    assert max_vec.shape == (sum(p.numel() for p in model.parameters()),)
    assert min_vec.shape == max_vec.shape
    assert iterations > 0

    max_only, none_min, *_ = min_max_hessian_eigs_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )

    assert isinstance(max_only, float)
    assert none_min is None
