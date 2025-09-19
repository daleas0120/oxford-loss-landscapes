"""
Hessian computation utilities for loss landscape analysis.
"""

from .hessian import (
    min_max_hessian_eigs,
    eval_hess_vec_prod,
    npvec_to_tensorlist,
    gradtensor_to_npvec,
    get_eigenstuff,
    get_hessian,
    small_hessian,
    hessian_vector_product,
    create_hessian_vector_product,
    create_hessian_vector_product_from_loss,
    get_hessian_eigenstuff
)
from .hessian_trace import hessian_trace
from .vrpca import (
    VRPCAConfig,
    VRPCAResult,
    min_max_hessian_eigs_vrpca,
    top_hessian_eigenpair_vrpca,
)

__all__ = [
    "min_max_hessian_eigs",
    "eval_hess_vec_prod",
    "hessian_trace",
    "npvec_to_tensorlist",
    "gradtensor_to_npvec",
    "VRPCAConfig",
    "VRPCAResult",
    "top_hessian_eigenpair_vrpca",
    "min_max_hessian_eigs_vrpca",
    "get_eigenstuff",
    "get_hessian",
    "small_hessian",
    "hessian_vector_product",
    "create_hessian_vector_product",
    "create_hessian_vector_product_from_loss",
    "get_hessian_eigenstuff"
]
