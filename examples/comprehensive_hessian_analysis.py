#!/usr/bin/env python3
"""
Comprehensive Hessian Analysis: A Demonstrative Hybrid Approach

This script performs a full Hessian analysis, demonstrating how the VR-PCA solver
can be used as a drop-in replacement for the maximum eigenvalue calculation in a
larger analysis pipeline, which is especially useful for large models.
"""

from __future__ import annotations

import argparse
import sys
import time
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
        if (epoch + 1) % (max(1, training_epochs // 5)) == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    # Fine-grained controls
    pgroup = parser.add_argument_group("Fine-grained controls")
    pgroup.add_argument("--samples", type=int, default=500, help="Total data samples.")
    pgroup.add_argument("--input-dim", type=int, default=20, help="Input dimension.")
    pgroup.add_argument("--hidden-dim", type=int, default=40, help="Hidden dimension.")

    # General options
    ogroup = parser.add_argument_group("General options")
    ogroup.add_argument("--training-epochs", type=int, default=30)
    ogroup.add_argument("--vrpca-epochs", type=int, default=15, help="VR-PCA epochs")
    ogroup.add_argument("--vrpca-batch-size", type=int, default=128, help="VR-PCA batch size")
    ogroup.add_argument("--vrpca-inner-loop-factor", type=float, default=0.1, help="VR-PCA inner loop factor.")
    ogroup.add_argument("--seed", type=int, default=42)
    ogroup.add_argument("--verbose-classical", action="store_true", help="Show verbose iteration output for the classical solver.")

    # Preset size controls
    sgroup = parser.add_mutually_exclusive_group()
    sgroup.add_argument("--small", action="store_true", help="Use small model/dataset preset (default)." )
    sgroup.add_argument("--medium", action="store_true", help="Use medium model/dataset preset.")
    sgroup.add_argument("--large", action="store_true", help="Use large model/dataset preset to see VR-PCA benefits.")

    args = parser.parse_args(argv)

    # --- Apply Presets ---
    print("Comprehensive Hessian Analysis: A Hybrid Approach")
    if args.medium:
        print("-> Using MEDIUM preset to show performance differences.")
        args.samples = 5000
        args.input_dim = 50
        args.hidden_dim = 100
    elif args.large:
        print("-> Using LARGE preset to clearly show VR-PCA benefits.")
        args.samples = 20000
        args.input_dim = 100
        args.hidden_dim = 200
    else: # small is the default
        print("-> Using SMALL preset (default for quick demo). Use --medium or --large to see performance benefits.")

    print("=" * 50)

    # Setup
    model = build_model(args.input_dim, args.hidden_dim)
    inputs, targets = make_data(args.samples, args.input_dim, args.seed)
    criterion = nn.MSELoss()
    print(f"Data samples: {args.samples}, Model parameters: {sum(p.numel() for p in model.parameters())}")
    train_model(model, inputs, targets, args.training_epochs)

    # --- Run All Analyses ---
    print("\nRunning all Hessian analyses...")
    vrpca_config = VRPCAConfig(
        batch_size=args.vrpca_batch_size, 
        epochs=args.vrpca_epochs, 
        seed=args.seed,
        inner_loop_factor=args.vrpca_inner_loop_factor
    )
    vrpca_result = top_hessian_eigenpair_vrpca(
        net=model, inputs=inputs, targets=targets, criterion=criterion, config=vrpca_config
    )
    vrpca_min_result = min_hessian_eigenpair_vrpca(
        net=model,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=vrpca_config,
    )
    
    start_time = time.perf_counter()
    max_eig, min_eig, _, _, iterations = min_max_hessian_eigs(
        model, inputs, targets, criterion, rank=0, use_cuda=False, verbose=args.verbose_classical, all_params=True
    )
    classical_elapsed_time = time.perf_counter() - start_time
    
    trace = hessian_trace(model, criterion, inputs, targets, num_random_vectors=10)
    print("✓ All analyses complete.")

    # --- Full Report ---
    print("\n\n--- Full Analysis Report ---")
    print("-" * 50)

    # VR-PCA summary for both extremes
    vrpca_total_time = vrpca_result.elapsed_time + vrpca_min_result.elapsed_time
    vrpca_total_cost = vrpca_result.hvp_equivalent_calls + vrpca_min_result.hvp_equivalent_calls
    dropin_max, dropin_min, _, _, dropin_cost = min_max_hessian_eigs(
        net=model,
        inputs=inputs,
        outputs=targets,
        criterion=criterion,
        backend="vrpca",
        vrpca_config=vrpca_config,
        compute_min=True,
    )
    rel = lambda est, ref: abs(est - ref) / max(abs(ref), 1e-8)

    print("\nSECTION 1: VR-PCA vs Classical Extremes")
    print("For large models, the classical method becomes slow. VR-PCA now covers both eigenpairs.")

    print("\n  [Classical Solver Result]")
    print(f"  - Max Eigenvalue : {max_eig:.6f} (baseline)")
    print(f"  - Min Eigenvalue : {min_eig:.6f} (baseline)")
    print(f"  - Wall Time      : {classical_elapsed_time:.3f}s")
    print(f"  - Cost           : {iterations} iterations (full HVPs)")

    print("\n  [VR-PCA Solver Result]")
    print(f"  - Max Eigenvalue : {vrpca_result.eigenvalue:.6f} (err {rel(vrpca_result.eigenvalue, max_eig):.2%})")
    print(f"  - Min Eigenvalue : {vrpca_min_result.eigenvalue:.6f} (err {rel(vrpca_min_result.eigenvalue, min_eig):.2%})")
    print(f"  - Wall Time      : {vrpca_total_time:.3f}s")
    print(f"  - Cost           : {vrpca_total_cost:.2f} HVP equivalents")
    print(f"  - Drop-in cost   : {dropin_cost:.2f} (from min_max_hessian_eigs backend)")

    print("\n\nSECTION 2: Remainder of the Analysis Pipeline")
    print("The curvature report below can rely on either classical or VR-PCA results.")

    print("\n  [Full Curvature Analysis]")
    print(f"  - Minimum Eigenvalue (VR-PCA) : {vrpca_min_result.eigenvalue:.6f}")
    print(f"  - Condition Number (VR-PCA)   : {abs(vrpca_result.eigenvalue / vrpca_min_result.eigenvalue):.2f}")
    if vrpca_result.eigenvalue > 0 and vrpca_min_result.eigenvalue < 0:
        print("  - Critical Point    : → Saddle point (indefinite Hessian)")
    elif vrpca_result.eigenvalue > 0 and vrpca_min_result.eigenvalue > 0:
        print("  - Critical Point    : → Local minimum (positive definite Hessian)")
    else:
        print("  - Critical Point    : → Other")

    print("\n  [Hessian Trace]")
    print(f"  - Estimated Trace   : {trace:.6f}")

    print("\n" + "=" * 50)
    print("✓ Report complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
