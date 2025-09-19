#!/usr/bin/env python3
"""Compare classical eigensolvers with VR-PCA on synthetic models."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oxford_loss_landscapes.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig, top_hessian_eigenpair_vrpca


@dataclass
class ExperimentResult:
    solver: str
    eigenvalue: float
    elapsed: float
    iterations: Optional[float]
    converged: Optional[bool] = None


def make_model(width: int, input_dim: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(input_dim, width),
        nn.Tanh(),
        nn.Linear(width, 1),
    )
    return model


def make_data(samples: int, input_dim: int, seed: int) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.standard_normal((samples, input_dim))).float()
    true_w = torch.from_numpy(rng.standard_normal((input_dim, 1))).float()
    y = X @ true_w + 0.1 * torch.randn(samples, 1)
    return {"inputs": X, "targets": y}


def run_eigsh(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion) -> ExperimentResult:
    start = time.perf_counter()
    max_eig, _, _, _, iterations = min_max_hessian_eigs(
        net=model,
        inputs=inputs,
        outputs=targets,
        criterion=criterion,
        rank=0,
        use_cuda=False,
        verbose=False,
        all_params=True,
    )
    elapsed = time.perf_counter() - start
    return ExperimentResult(
        solver="eigsh",
        eigenvalue=float(max_eig),
        elapsed=elapsed,
        iterations=iterations,
        converged=True,
    )


def run_vrpca(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion, config: VRPCAConfig) -> ExperimentResult:
    start = time.perf_counter()
    result = top_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )
    elapsed = time.perf_counter() - start
    return ExperimentResult(
        solver="vrpca",
        eigenvalue=float(result.eigenvalue),
        elapsed=elapsed,
        iterations=result.hvp_equivalent_calls,
        converged=result.converged,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=10, help="hidden layer width")
    parser.add_argument("--input-dim", type=int, default=10, help="input dimensionality")
    parser.add_argument("--samples", type=int, default=128, help="number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=10, help="VR-PCA epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="VR-PCA minibatch size")
    parser.add_argument("--solver", choices=["eigsh", "vrpca", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    model = make_model(args.width, args.input_dim)
    data = make_data(args.samples, args.input_dim, args.seed)
    criterion = nn.MSELoss()

    if args.solver in {"eigsh", "both"}:
        result = run_eigsh(model, data["inputs"], data["targets"], criterion)
        print(
            "eigsh: eigenvalue={:.6f}, elapsed={:.3f}s, iterations={}".format(
                result.eigenvalue,
                result.elapsed,
                result.iterations,
            )
        )

    if args.solver in {"vrpca", "both"}:
        config = VRPCAConfig(batch_size=args.batch_size, epochs=args.epochs)
        result = run_vrpca(model, data["inputs"], data["targets"], criterion, config=config)
        print(
            "vrpca: eigenvalue={:.6f}, elapsed={:.3f}s, hvp_equiv={:.2f}, converged={}".format(
                result.eigenvalue,
                result.elapsed,
                float(result.iterations or 0.0),
                "yes" if result.converged else "no",
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
