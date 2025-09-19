"""Hessian computation utilities for loss landscape analysis."""

from __future__ import annotations

from typing import Optional

from .hessian_trace import hessian_trace
from .vrpca import (
    VRPCAConfig,
    VRPCAResult,
    min_hessian_eigenpair_vrpca,
    min_max_hessian_eigs_vrpca,
    top_hessian_eigenpair_vrpca,
)

CLASSICAL_AVAILABLE = True
_CLASSICAL_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - exercised indirectly via classical tests
    from .hessian import (
        min_max_hessian_eigs,
        eval_hess_vec_prod,
        npvec_to_tensorlist,
        gradtensor_to_npvec,
    )
except Exception as exc:  # pragma: no cover - depends on SciPy availability
    CLASSICAL_AVAILABLE = False
    _CLASSICAL_IMPORT_ERROR = exc

    def _raise_classical_import(name: str):  # type: ignore[override]
        raise ImportError(
            f"Classical Hessian eigensolvers require SciPy. Install 'scipy' to use {name}."
        ) from _CLASSICAL_IMPORT_ERROR

    def min_max_hessian_eigs(*args, **kwargs):  # type: ignore[override]
        _raise_classical_import("min_max_hessian_eigs")

    def eval_hess_vec_prod(*args, **kwargs):  # type: ignore[override]
        _raise_classical_import("eval_hess_vec_prod")

    def npvec_to_tensorlist(*args, **kwargs):  # type: ignore[override]
        _raise_classical_import("npvec_to_tensorlist")

    def gradtensor_to_npvec(*args, **kwargs):  # type: ignore[override]
        _raise_classical_import("gradtensor_to_npvec")

__all__ = [
    "CLASSICAL_AVAILABLE",
    "min_max_hessian_eigs",
    "eval_hess_vec_prod",
    "hessian_trace",
    "npvec_to_tensorlist",
    "gradtensor_to_npvec",
    "VRPCAConfig",
    "VRPCAResult",
    "top_hessian_eigenpair_vrpca",
    "min_hessian_eigenpair_vrpca",
    "min_max_hessian_eigs_vrpca",
]
