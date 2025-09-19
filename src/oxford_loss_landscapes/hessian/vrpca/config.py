"""Configuration objects for VR-PCA based Hessian routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VRPCAConfig:
    """Runtime settings for VR-PCA Hessian eigensolvers."""

    batch_size: int = 64
    epochs: int = 20
    tol: float = 1e-4
    eta_mode: str = "adaptive"
    eta_fixed: float = 1e-2
    seed: int = 0
    use_minibatch_equivalent: bool = True
    budget: Optional[float] = None
    track_window: int = 5

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.eta_mode not in {"adaptive", "fixed"}:
            raise ValueError("eta_mode must be 'adaptive' or 'fixed'")
        if self.eta_fixed <= 0:
            raise ValueError("eta_fixed must be positive")
        if self.track_window <= 0:
            raise ValueError("track_window must be positive")


@dataclass
class VRPCAResult:
    """Primary outputs from a VR-PCA eigenpair computation."""

    eigenvalue: float
    eigenvector: "torch.Tensor"
    hvp_equivalent_calls: float
    epochs_completed: int
    converged: bool
    elapsed_time: float

    def normalized_vector(self) -> "torch.Tensor":
        """Return the stored eigenvector renormalized to unit length."""
        import torch

        vec = self.eigenvector
        norm = vec.norm().clamp_min(torch.finfo(vec.dtype).eps)
        return vec / norm
