"""Public API entry points for VR-PCA Hessian solvers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import numpy as np

from .algorithms import top_eigenpair_vrpca
from .config import VRPCAConfig, VRPCAResult
from .hvp import HessianVectorProductOracle


def _resolve_device(module: torch.nn.Module, use_cuda: bool) -> torch.device:
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return next(module.parameters()).device


def _move_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return module.to(device=device)


def _ensure_tensor_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    if isinstance(data, (list, tuple)):
        converted = [_ensure_tensor_device(item, device) for item in data]
        return type(data)(converted) if isinstance(data, tuple) else converted
    raise TypeError("inputs and targets must be tensors or sequences of tensors")


def top_hessian_eigenpair_vrpca(
    net: torch.nn.Module,
    inputs,
    targets,
    criterion,
    *,
    all_params: bool = True,
    use_cuda: bool = False,
    config: Optional[VRPCAConfig] = None,
) -> VRPCAResult:
    """Compute the dominant Hessian eigenpair using VR-PCA."""

    device = _resolve_device(net, use_cuda)
    net = _move_module(net, device)
    inputs = _ensure_tensor_device(inputs, device)
    targets = _ensure_tensor_device(targets, device)

    oracle = HessianVectorProductOracle(
        model=net,
        loss_fn=criterion,
        inputs=inputs,
        targets=targets,
        all_params=all_params,
    )
    return top_eigenpair_vrpca(oracle, config=config)


def min_max_hessian_eigs_vrpca(
    net: torch.nn.Module,
    inputs,
    targets,
    criterion,
    *,
    all_params: bool = True,
    use_cuda: bool = False,
    config: Optional[VRPCAConfig] = None,
) -> Tuple[float, Optional[float], "np.ndarray", Optional[torch.Tensor], float]:
    """Return an interface similar to ``min_max_hessian_eigs`` using VR-PCA for the top eigenpair.

    The minimum eigen-information is not currently provided by VR-PCA. To keep the return
    signature compatible with the classical helper, this function returns ``None`` for the
    minimum eigenvalue/vector and the classical iteration counter.
    """

    result = top_hessian_eigenpair_vrpca(
        net=net,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        all_params=all_params,
        use_cuda=use_cuda,
        config=config,
    )
    return (
        result.eigenvalue,
        None,
        result.eigenvector.detach().cpu().numpy(),
        None,
        result.hvp_equivalent_calls,
    )
