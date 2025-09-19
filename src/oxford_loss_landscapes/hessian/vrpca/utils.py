"""Utility helpers for VR-PCA Hessian solvers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from .types import TensorList, TensorSequence


def collect_parameters(model: torch.nn.Module, all_params: bool) -> List[torch.nn.Parameter]:
    """Return parameters to include in Hessian operations."""
    selected: List[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if all_params or param.dim() > 1:
            selected.append(param)
    if not selected:
        raise ValueError("model has no differentiable parameters for Hessian computation")
    return selected


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Concatenate a collection of tensors into a 1D view."""
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def unflatten_like(vector: torch.Tensor, reference: Sequence[torch.Tensor]) -> TensorList:
    """Reshape a flat vector into tensors shaped like the reference sequence."""
    outputs: List[torch.Tensor] = []
    index = 0
    for ref in reference:
        numel = ref.numel()
        outputs.append(vector[index : index + numel].view_as(ref))
        index += numel
    if index != vector.numel():
        raise ValueError("vector size does not match parameter shapes")
    return outputs


def normalize_vector(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a unit-norm copy of the provided vector."""
    norm = vector.norm()
    if float(norm) <= eps:
        return vector.clone()
    return vector / norm


def rayleigh_quotient(vector: torch.Tensor, hvp: TensorList) -> float:
    """Compute the Rayleigh quotient for the provided vector and HVP."""
    numerator = torch.dot(vector, flatten_tensors(hvp))
    denominator = torch.dot(vector, vector).clamp_min(1e-12)
    return float(numerator / denominator)


def ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to the requested device if required."""
    if tensor.device == device:
        return tensor
    return tensor.to(device=device)


def ensure_batch_on_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Ensure mini-batch inputs/targets reside on the correct device."""
    return ensure_device(batch, device)


def with_no_grad_tensors(tensors: TensorSequence) -> TensorList:
    """Detach tensors to avoid storing graphs while preserving device information."""
    return [tensor.detach() for tensor in tensors]
