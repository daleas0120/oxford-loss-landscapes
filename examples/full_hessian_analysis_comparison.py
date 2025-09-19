#!/usr/bin/env python3
"""
Full Hessian Analysis Comparison: Classical vs. VR-PCA

This script demonstrates a comprehensive Hessian analysis, comparing the
classical `eigsh`-based solver with the stochastic VR-PCA solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow running the example from a source checkout without installation
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oxford_loss_landscapes.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace
from oxford_loss_landscapes.hessian.vrpca import (
    VRPCAConfig,
    min_hessian_eigenpair_vrpca,
    top_hessian_eigenpair_vrpca,
)


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    """Creates a simple two-layer MLP."""
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

def make_data(samples: int, input_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic data for a simple regression problem."""
    torch.manual_seed(seed)
    inputs = torch.randn(samples, input_dim)
    weights = torch.randn(input_dim, 1)
    targets = inputs @ weights + 0.1 * torch.randn(samples, 1)
    return inputs, targets

def train_model(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, training_epochs: int) -> None:
    """Quickly trains the model on the provided data."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("\nTraining model...")
    for epoch in range(training_epochs):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--training-epochs", type=int, default=50)
    parser.add_argument("--vrpca-epochs", type=int, default=15, help="VR-PCA epochs")
    parser.add_argument("--vrpca-batch-size", type=int, default=64, help="VR-PCA batch size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print("Full Hessian Analysis: Classical vs. VR-PCA")
    print("=" * 50)

    # Setup model, data, and loss
    model = build_model(args.input_dim, args.hidden)
    inputs, targets = make_data(args.samples, args.input_dim, args.seed)
    criterion = nn.MSELoss()

    print(f"Data shape: X={inputs.shape}, y={targets.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train the model
    train_model(model, inputs, targets, args.training_epochs)

    # --- VR-PCA Analysis ---
    print("\n\n--- Running VR-PCA Solver ---")
    vrpca_config = VRPCAConfig(batch_size=args.vrpca_batch_size, epochs=args.vrpca_epochs, seed=args.seed)
    vrpca_result = top_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=vrpca_config,
    )
    vrpca_min_result = min_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=vrpca_config,
    )
    dropin_max, dropin_min, _, _, dropin_cost = min_max_hessian_eigs(
        net=model,
        inputs=inputs,
        outputs=targets,
        criterion=criterion,
        backend="vrpca",
        vrpca_config=vrpca_config,
        compute_min=True,
    )
    print("✓ VR-PCA analysis complete.")
    print("\n--- Running Classical (eigsh) Solver ---")
    max_eig, min_eig, max_vec, min_vec, iterations = min_max_hessian_eigs(
        model, inputs, targets, criterion, rank=0, use_cuda=False, verbose=True, all_params=True
    )
    print("✓ Classical analysis complete.")

    # --- Combined Report ---
    print("\n\n--- Full Analysis Report ---")
    print("-" * 50)

    # VR-PCA Results
    print("\n[VR-PCA Solver Results]")
    vrpca_total_time = vrpca_result.elapsed_time + vrpca_min_result.elapsed_time
    vrpca_total_cost = vrpca_result.hvp_equivalent_calls + vrpca_min_result.hvp_equivalent_calls
    rel = lambda est, ref: abs(est - ref) / max(abs(ref), 1e-8)
    print(f"Max Eigenvalue      : {vrpca_result.eigenvalue:.6f} (err {rel(vrpca_result.eigenvalue, max_eig):.2%})")
    print(f"Min Eigenvalue      : {vrpca_min_result.eigenvalue:.6f} (err {rel(vrpca_min_result.eigenvalue, min_eig):.2%})")
    print(f"Elapsed Time (sum)  : {vrpca_total_time:.3f}s")
    print(f"HVP Equivalents     : {vrpca_total_cost:.2f} (drop-in reports {dropin_cost:.2f})")
    print(f"Max converged       : {vrpca_result.converged}")
    print(f"Min converged       : {vrpca_min_result.converged}")
    print(f"Max vector norm     : {vrpca_result.eigenvector.norm():.6f}")
    print(f"Min vector norm     : {vrpca_min_result.eigenvector.norm():.6f}")

    # Classical Results & Combined Analysis
    print("\n[Classical Solver Results & Combined Analysis]")
    print(f"Maximum Eigenvalue  : {max_eig:.6f}")
    print(f"Minimum Eigenvalue  : {min_eig:.6f}")
    print(f"Condition Number    : {abs(max_eig / min_eig):.2f}")
    print(f"Iterations          : {iterations}")

    if max_eig > 0 and min_eig < 0:
        print("Critical Point      : → Saddle point (indefinite Hessian)")
    elif max_eig > 0 and min_eig > 0:
        print("Critical Point      : → Local minimum (positive definite Hessian)")
    else:
        print("Critical Point      : → Other")

    print("\n[Eigenvector Properties (from Classical Solver)]")
    print(f"Max Eigenvector Shape: {max_vec.shape}")
    print(f"Min Eigenvector Shape: {min_vec.shape}")
    print(f"Max Eigenvector Norm : {np.linalg.norm(max_vec):.6f}")
    print(f"Min Eigenvector Norm : {np.linalg.norm(min_vec):.6f}")

    # Hessian Trace
    print("\n[Hessian Trace Estimation]")
    trace = hessian_trace(model, criterion, inputs, targets, num_random_vectors=10)
    print(f"Estimated Trace     : {trace:.6f}")

    print("\n" + "=" * 50)
    print("✓ All analyses completed successfully!")


if __name__ == "__main__":  # pragma: no cover
    main()
