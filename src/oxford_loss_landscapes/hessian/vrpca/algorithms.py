"""Algorithms for VR-PCA based Hessian eigen-computation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import VRPCAConfig, VRPCAResult
from .hvp import HVPBudgetTracker, HessianVectorProductOracle
from .types import TensorList
from .utils import (
    flatten_tensors,
    normalize_vector,
    rayleigh_quotient,
    unflatten_like,
    with_no_grad_tensors,
)


@dataclass
class _VRPCAState:
    vector: torch.Tensor
    snapshot: torch.Tensor


class ConvergenceTracker:
    """Track eigenvalue convergence over the optimisation."""

    def __init__(self, window: int, tol: float) -> None:
        self.window = window
        self.tol = tol
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.values.append(value)

    def converged(self) -> bool:
        if len(self.values) < self.window:
            return False
        recent = self.values[-self.window :]
        mean_value = sum(recent) / float(len(recent))
        if math.isclose(mean_value, 0.0, abs_tol=1e-12):
            spread = max(recent) - min(recent)
            return spread < self.tol
        deviation = (max(recent) - min(recent)) / abs(mean_value)
        return deviation < self.tol


def _initialise_state(oracle: HessianVectorProductOracle, config: VRPCAConfig) -> _VRPCAState:
    generator = torch.Generator(device=oracle.device)
    generator.manual_seed(config.seed)
    base = torch.randn(
        oracle.num_parameters,
        device=oracle.device,
        dtype=oracle.params[0].dtype,
        generator=generator,
    )
    vector = normalize_vector(base)
    return _VRPCAState(vector=vector, snapshot=vector.clone())


def _flatten_hvp(tensors: TensorList) -> torch.Tensor:
    return flatten_tensors(with_no_grad_tensors(tensors))


def _inner_iterations(config: VRPCAConfig, dataset_size: int) -> int:
    if config.budget is None:
        target = max(1, int(config.inner_loop_factor * dataset_size))
        return min(target, dataset_size)
    per_epoch_budget = max(config.budget / float(config.epochs), 1.0)
    minibatch_cost = 2.0 * float(config.batch_size) / float(max(dataset_size, 1))
    effective_cost = max(minibatch_cost, 1e-12)
    inner = int(max((per_epoch_budget - 1.0) / effective_cost, 1.0))
    return min(inner, dataset_size)


def top_eigenpair_vrpca(
    oracle: HessianVectorProductOracle,
    config: Optional[VRPCAConfig] = None,
) -> VRPCAResult:
    config = config or VRPCAConfig()
    tracker = ConvergenceTracker(window=config.track_window, tol=config.tol)
    budget = HVPBudgetTracker()

    state = _initialise_state(oracle, config)
    dataset_size = oracle.dataset_size
    if dataset_size <= 0:
        raise ValueError("dataset must contain at least one sample for VR-PCA")
    inner_steps = _inner_iterations(config, dataset_size)

    if config.eta_mode == "adaptive":
        step_size = oracle.adaptive_step_size(default=config.eta_fixed)
    else:
        step_size = config.eta_fixed

    rng = np.random.default_rng(config.seed)
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        snapshot_list = unflatten_like(state.snapshot, oracle.params)
        snapshot_hvp = oracle.full_hvp(snapshot_list, tracker=budget)
        snapshot_hvp_flat = _flatten_hvp(snapshot_hvp)

        vector = state.vector.clone()

        for _ in range(inner_steps):
            indices = rng.integers(0, dataset_size, size=min(config.batch_size, dataset_size))
            vec_list = unflatten_like(vector, oracle.params)
            snap_list = unflatten_like(state.snapshot, oracle.params)

            hvp_vec = oracle.minibatch_hvp(vec_list, indices, tracker=budget)
            hvp_snap = oracle.minibatch_hvp(snap_list, indices, tracker=budget)

            update = _flatten_hvp(hvp_vec) - _flatten_hvp(hvp_snap) + snapshot_hvp_flat
            vector = normalize_vector(vector + step_size * update)

        state = _VRPCAState(vector=vector, snapshot=vector.clone())

        eigvec_list = unflatten_like(vector, oracle.params)
        eig_hvp = oracle.full_hvp(eigvec_list, tracker=budget)
        eigval = rayleigh_quotient(vector, with_no_grad_tensors(eig_hvp))
        tracker.add(eigval)

        if tracker.converged():
            elapsed = time.perf_counter() - start
            return VRPCAResult(
                eigenvalue=eigval,
                eigenvector=vector,
                hvp_equivalent_calls=budget.total_equivalent(),
                epochs_completed=epoch,
                converged=True,
                elapsed_time=elapsed,
            )

    elapsed = time.perf_counter() - start
    final_vec = state.vector
    eigvec_list = unflatten_like(final_vec, oracle.params)
    eig_hvp = oracle.full_hvp(eigvec_list, tracker=budget)
    eigval = rayleigh_quotient(final_vec, with_no_grad_tensors(eig_hvp))
    return VRPCAResult(
        eigenvalue=eigval,
        eigenvector=final_vec,
        hvp_equivalent_calls=budget.total_equivalent(),
        epochs_completed=config.epochs,
        converged=False,
        elapsed_time=elapsed,
    )


def top_eigenpair_oja(
    oracle: HessianVectorProductOracle,
    iterations: int,
    step_scale: float = 0.01,
    seed: int = 0,
) -> VRPCAResult:
    generator = torch.Generator(device=oracle.device)
    generator.manual_seed(seed)
    vector = torch.randn(
        oracle.num_parameters,
        device=oracle.device,
        dtype=oracle.params[0].dtype,
        generator=generator,
    )
    vector = normalize_vector(vector)
    budget = HVPBudgetTracker()
    rng = np.random.default_rng(seed)
    dataset_size = oracle.dataset_size
    if dataset_size <= 0:
        raise ValueError("dataset must contain at least one sample for VR-PCA")
    start = time.perf_counter()

    for step in range(1, iterations + 1):
        indices = rng.integers(0, dataset_size, size=min(dataset_size, 512))
        vec_list = unflatten_like(vector, oracle.params)
        hvp_vec = oracle.minibatch_hvp(vec_list, indices, tracker=budget)
        hvp_flat = _flatten_hvp(hvp_vec)
        eta = step_scale / float(step)
        vector = normalize_vector(vector + eta * hvp_flat)

    elapsed = time.perf_counter() - start
    eigvec_list = unflatten_like(vector, oracle.params)
    eig_hvp = oracle.full_hvp(eigvec_list, tracker=budget)
    eigval = rayleigh_quotient(vector, with_no_grad_tensors(eig_hvp))
    return VRPCAResult(
        eigenvalue=eigval,
        eigenvector=vector,
        hvp_equivalent_calls=budget.total_equivalent(),
        epochs_completed=iterations,
        converged=True,
        elapsed_time=elapsed,
    )
