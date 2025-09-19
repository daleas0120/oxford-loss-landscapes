"""Hessian-vector product utilities for VR-PCA solvers."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from .types import TensorList, TensorSequence
from .utils import (
    collect_parameters,
    flatten_tensors,
    normalize_vector,
    unflatten_like,
    with_no_grad_tensors,
)


def _apply_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        converted = [_apply_to_device(item, device) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    raise TypeError("inputs and targets must be tensors or sequences of tensors")


def _select_batch(data, indices: Optional[Sequence[int]]):
    if indices is None:
        return data
    if isinstance(data, torch.Tensor):
        return data[indices]
    if isinstance(data, (list, tuple)):
        sliced = [_select_batch(item, indices) for item in data]
        return type(data)(sliced) if isinstance(data, tuple) else sliced
    raise TypeError("inputs and targets must be tensors or sequences of tensors")


class HVPBudgetTracker:
    """Keep track of full and minibatch Hessian-vector products."""

    def __init__(self) -> None:
        self.full_calls = 0
        self.minibatch_calls = 0
        self.minibatch_equivalent = 0.0

    def add_full(self, count: int = 1) -> None:
        self.full_calls += count

    def add_minibatch(self, batch_size: int, total_size: int) -> None:
        total_size = max(total_size, 1)
        self.minibatch_calls += 1
        self.minibatch_equivalent += float(batch_size) / float(total_size)

    def total_equivalent(self) -> float:
        return float(self.full_calls) + float(self.minibatch_equivalent)


class HessianVectorProductOracle:
    """Callable Hessian-vector products for a model/loss pair."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs,
        targets,
        *,
        all_params: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = next(model.parameters()).device
        self.inputs = _apply_to_device(inputs, self.device)
        self.targets = _apply_to_device(targets, self.device)
        self.params = collect_parameters(model, all_params)
        self._spectral_norm_estimate: Optional[float] = None

    @property
    def num_parameters(self) -> int:
        return int(sum(param.numel() for param in self.params))

    @property
    def dataset_size(self) -> int:
        if isinstance(self.inputs, torch.Tensor):
            return int(self.inputs.shape[0])
        if isinstance(self.inputs, (list, tuple)) and self.inputs:
            first = self.inputs[0]
            if isinstance(first, torch.Tensor):
                return int(first.shape[0])
        raise TypeError("inputs must be a tensor or sequence of tensors with a batch dimension")

    def hvp(self, vector_list: TensorSequence, batch_indices: Optional[Sequence[int]] = None) -> TensorList:
        xb = _apply_to_device(_select_batch(self.inputs, batch_indices), self.device)
        yb = _apply_to_device(_select_batch(self.targets, batch_indices), self.device)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(xb)
        loss = self.loss_fn(outputs, yb)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)

        vector_list = [vec.to(device=self.device) for vec in vector_list]
        hvp = torch.autograd.grad(grads, self.params, grad_outputs=vector_list, retain_graph=False)
        return list(hvp)

    def full_hvp(self, vector_list: TensorSequence, tracker: Optional[HVPBudgetTracker] = None) -> TensorList:
        if tracker is not None:
            tracker.add_full()
        return self.hvp(vector_list, batch_indices=None)

    def minibatch_hvp(self, vector_list: TensorSequence, indices: Sequence[int], tracker: Optional[HVPBudgetTracker] = None) -> TensorList:
        if tracker is not None:
            tracker.add_minibatch(batch_size=len(indices), total_size=self.dataset_size)
        return self.hvp(vector_list, batch_indices=indices)

    def estimate_spectral_norm(self, power_iters: int = 5) -> float:
        vector = torch.randn(self.num_parameters, device=self.device, dtype=self.params[0].dtype)
        vector = normalize_vector(vector)
        norm_estimate = 0.0
        for _ in range(power_iters):
            vector_list = unflatten_like(vector, self.params)
            hvp = self.hvp(vector_list)
            hvp_flat = flatten_tensors(with_no_grad_tensors(hvp))
            norm_estimate = float(hvp_flat.norm().item())
            vector = normalize_vector(hvp_flat)
        self._spectral_norm_estimate = norm_estimate
        return norm_estimate

    def adaptive_step_size(self, default: float = 1e-2) -> float:
        if self._spectral_norm_estimate is None or math.isclose(self._spectral_norm_estimate, 0.0):
            self.estimate_spectral_norm()
        if self._spectral_norm_estimate is None or self._spectral_norm_estimate <= 0:
            return default
        return 0.5 / float(self._spectral_norm_estimate)
