#!/usr/bin/env python3
"""Example showing how to run the VR-PCA Hessian solver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running the example from a source checkout without installation
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oxford_loss_landscapes.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig, top_hessian_eigenpair_vrpca


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))


def make_data(samples: int, input_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    inputs = torch.randn(samples, input_dim)
    weights = torch.randn(input_dim, 1)
    targets = inputs @ weights + 0.05 * torch.randn(samples, 1)
    return inputs, targets


def train_model(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 50) -> None:
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimiser.step()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12, help="VR-PCA epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="VR-PCA batch size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="also run the classical eigsh solver")
    args = parser.parse_args(argv)

    model = build_model(args.input_dim, args.hidden)
    inputs, targets = make_data(args.samples, args.input_dim, args.seed)
    criterion = nn.MSELoss()

    train_model(model, inputs, targets)

    config = VRPCAConfig(batch_size=args.batch_size, epochs=args.epochs)
    result = top_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )

    print("VR-PCA dominant eigenpair")
    print("-------------------------")
    print(f"Eigenvalue      : {result.eigenvalue:.6f}")
    print(f"HVP equivalents : {result.hvp_equivalent_calls:.2f}")
    print(f"Converged       : {result.converged}")

    if args.compare:
        max_eig, min_eig, *_ = min_max_hessian_eigs(
            net=model,
            inputs=inputs,
            outputs=targets,
            criterion=criterion,
            rank=0,
            use_cuda=False,
            verbose=False,
            all_params=True,
        )
        print("\nClassical eigsh comparison")
        print("--------------------------")
        print(f"Max eigenvalue  : {max_eig:.6f}")
        print(f"Min eigenvalue  : {min_eig:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
