#!/usr/bin/env python3
"""Guide to the VR-PCA Hessian eigen-solver available in this repository."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running directly from a source checkout without installing the package
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oxford_loss_landscapes.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig, top_hessian_eigenpair_vrpca


def explain_vrpca_concepts() -> None:
    print("VR-PCA HESSIAN ANALYSIS GUIDE")
    print("=" * 60)
    print()
    print("Variance-Reduced PCA (VR-PCA) accelerates eigenvector extraction by combining")
    print("full-batch and mini-batch Hessian-vector products (HVPs).")
    print("Key ingredients:")
    print("• Snapshot vector: anchor computed with a full HVP.")
    print("• Mini-batch HVPs: cheap updates on random subsets of the data.")
    print("• Variance reduction: subtract/add the snapshot response to stabilise updates.")
    print()
    print("Why it helps:")
    print("• Classical eigsh needs O(N²) memory/time for dense Hessians.")
    print("• VR-PCA scales with the cost of HVPs and tolerates large models/datasets.")
    print("• The algorithm converges when the Rayleigh quotient stabilises.")


def build_toy_problem(seed: int = 123) -> tuple[nn.Module, torch.Tensor, torch.Tensor, nn.Module]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 1))
    inputs = torch.randn(160, 6)
    targets = torch.randn(160, 1)
    criterion = nn.MSELoss()
    return model, inputs, targets, criterion


def train_quickly(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> None:
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(40):
        optimiser.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimiser.step()


def run_vrpca_demo(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> None:
    print("\nRunning VR-PCA demo")
    print("-" * 40)
    config = VRPCAConfig(batch_size=64, epochs=10)
    result = top_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=config,
    )
    print(f"Eigenvalue estimate : {result.eigenvalue:.6f}")
    print(f"Converged           : {result.converged}")
    print(f"HVP equiv. calls    : {result.hvp_equivalent_calls:.2f}")
    print(f"Epochs completed    : {result.epochs_completed}")


def compare_with_eigsh(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> None:
    print("\nComparing with classical eigsh baseline")
    print("-" * 40)
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
    print(f"eigsh max eigenvalue : {max_eig:.6f}")
    print(f"eigsh min eigenvalue : {min_eig:.6f}")


def interpret_results() -> None:
    print("\nInterpreting outputs")
    print("-" * 40)
    print("• VR-PCA focuses on the dominant eigenpair. Use it when the top curvature matters.")
    print("• The HVP equivalent counter lets you compare cost against full-batch eigsh.")
    print("• If `converged` is False, increase epochs or adjust `inner_loop_factor`.")
    print("• To approximate the minimum eigenvalue, fall back to the classical solver.")


def main() -> None:
    explain_vrpca_concepts()
    model, inputs, targets, criterion = build_toy_problem()
    train_quickly(model, inputs, targets, criterion)
    run_vrpca_demo(model, inputs, targets, criterion)
    compare_with_eigsh(model, inputs, targets, criterion)
    interpret_results()


if __name__ == "__main__":  # pragma: no cover
    main()
