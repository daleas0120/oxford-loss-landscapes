#!/usr/bin/env python3
"""
Improved Hessian top-eigen computation: eigsh baseline vs VR-PCA
Focuses on VR-PCA with optional Oja algorithm
"""

import math
import os
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class AlgorithmConfig:
    """Configuration for algorithm parameters"""
    # Eigsh parameters
    eigsh_tol: float = 1e-3
    eigsh_maxiter: Optional[int] = None
    
    # VR-PCA parameters
    vr_eta_mode: str = "adaptive"  # "adaptive" or "fixed"
    vr_eta_fixed: float = 0.01
    vr_epochs_max: int = 20  # Maximum epochs
    vr_m_factor: float = 1.0  # REDUCED from 5.0 - m = factor * n
    
    # Common parameters
    batch_size: int = 32
    conv_tol: float = 1e-4
    
    # Monitoring
    track_convergence: bool = True
    verbose: bool = False


# [Keep all utility functions from original: params_list, flatten_params, unflatten_like, normalize_vec, make_model, make_data]
# ... (same as in your original file) ...

def params_list(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]

def flatten_params(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])

def unflatten_like(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    outs = []
    idx = 0
    for p in params:
        numel = p.numel()
        outs.append(vec[idx:idx+numel].view_as(p))
        idx += numel
    return outs

def normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.norm()
    return v if float(n) <= eps else v / n

def make_model(N: int, d_in: int = 10, device: str = "cpu", dtype: torch.dtype = torch.float32) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(d_in, N),
        nn.Tanh(),
        nn.Linear(N, 1)
    )
    return model.to(device=device, dtype=dtype)

def make_data(n_samples: int, d_in: int = 10, seed: int = 0,
              device: str = "cpu", dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    X = torch.from_numpy(rng.randn(n_samples, d_in)).to(device=device, dtype=dtype)
    true_w = torch.from_numpy(rng.randn(d_in, 1)).to(device=device, dtype=dtype)
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1, device=device, dtype=dtype)
    return X, y


# ---------------------------
# HVP accounting and operations (keep from original)
# ---------------------------

class HvpCounter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.full_calls = 0
        self.minibatch_calls = 0
        self.minibatch_equiv = 0.0
    
    def add_full(self, count: int = 1):
        self.full_calls += count
    
    def add_minibatch(self, bsz: int, n: int):
        self.minibatch_calls += 1
        self.minibatch_equiv += float(bsz) / float(max(1, n))
    
    def total_equiv(self) -> float:
        return float(self.full_calls) + float(self.minibatch_equiv)


@dataclass
class ConvergenceTracker:
    """Track convergence metrics during optimization"""
    eigenvalues: List[float] = field(default_factory=list)
    hvp_counts: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    relative_errors: List[float] = field(default_factory=list)
    
    def add(self, eigenvalue: float, hvp_count: float, time: float, ref_eigenvalue: Optional[float] = None):
        self.eigenvalues.append(eigenvalue)
        self.hvp_counts.append(hvp_count)
        self.times.append(time)
        if ref_eigenvalue is not None:
            rel_err = abs(eigenvalue - ref_eigenvalue) / (abs(ref_eigenvalue) + 1e-12)
            self.relative_errors.append(rel_err)
    
    def check_convergence(self, tol: float, window: int = 5) -> bool:
        if len(self.eigenvalues) < window:
            return False
        recent = self.eigenvalues[-window:]
        return np.std(recent) / (abs(np.mean(recent)) + 1e-12) < tol


class HvpOps:
    def __init__(self, model: nn.Module, loss_fn, X: torch.Tensor, y: torch.Tensor):
        self.model = model
        self.loss_fn = loss_fn
        self.X = X
        self.y = y
        self.params = params_list(model)
        self._norm_estimate = None
        self._estimate_spectral_norm()
    
    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.params)
    
    def _estimate_spectral_norm(self, power_iters: int = 5):
        P = self.num_params()
        device = self.params[0].device
        dtype = self.params[0].dtype
        
        v = torch.randn(P, device=device, dtype=dtype)
        v = normalize_vec(v)
        
        for _ in range(power_iters):
            v_list = unflatten_like(v, self.params)
            hv_list = self.hvp_for_batch(v_list, batch_idx=None)
            hv_flat = flatten_params([h.detach() for h in hv_list])
            norm = hv_flat.norm().item()
            v = normalize_vec(hv_flat)
        
        self._norm_estimate = norm
    
    def get_adaptive_stepsize(self) -> float:
        if self._norm_estimate is None or self._norm_estimate <= 0:
            return 0.01
        return 0.5 / self._norm_estimate
    
    def hvp_for_batch(self, v_list: List[torch.Tensor], batch_idx: Optional[np.ndarray] = None) -> List[torch.Tensor]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        
        if batch_idx is None:
            xb = self.X
            yb = self.y
        else:
            xb = self.X[batch_idx]
            yb = self.y[batch_idx]
        
        out = self.model(xb)
        loss = self.loss_fn(out, yb)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)
        hvp = torch.autograd.grad(grads, self.params, grad_outputs=v_list, retain_graph=False)
        return hvp


def hvp_full_counted(hvp_ops: HvpOps, counter: HvpCounter, v_list: List[torch.Tensor]) -> List[torch.Tensor]:
    counter.add_full(1)
    return hvp_ops.hvp_for_batch(v_list, batch_idx=None)

def hvp_minibatch_counted(hvp_ops: HvpOps, counter: HvpCounter, v_list: List[torch.Tensor], batch_idx: np.ndarray) -> List[torch.Tensor]:
    counter.add_minibatch(bsz=int(batch_idx.shape[0]), n=int(hvp_ops.X.shape[0]))
    return hvp_ops.hvp_for_batch(v_list, batch_idx=batch_idx)

def rayleigh_quotient(hvp_ops: HvpOps, w_flat: torch.Tensor) -> float:
    params = hvp_ops.params
    v_list = unflatten_like(w_flat, params)
    hvp_list = hvp_ops.hvp_for_batch(v_list, batch_idx=None)
    hvp_flat = flatten_params([g.detach() for g in hvp_list])
    num = (w_flat * hvp_flat).sum().item()
    den = (w_flat.norm().pow(2).item() + 1e-12)
    return float(num / den)


# ---------------------------
# Algorithm implementations
# ---------------------------

def top_hessian_eig_eigsh(
    hvp_ops: HvpOps, 
    counter: HvpCounter, 
    config: AlgorithmConfig,
    tracker: Optional[ConvergenceTracker] = None
) -> Tuple[float, torch.Tensor]:
    """Compute top eigenvalue using eigsh"""
    params = hvp_ops.params
    P = hvp_ops.num_params()
    
    def matvec(vec_np: np.ndarray) -> np.ndarray:
        vec = torch.from_numpy(vec_np.astype(np.float32, copy=False)).to(
            device=params[0].device, dtype=params[0].dtype
        )
        v_list = unflatten_like(vec, params)
        hvp_list = hvp_full_counted(hvp_ops, counter, v_list)
        hvp_flat = flatten_params([h.detach() for h in hvp_list]).to("cpu").numpy().astype(np.float64, copy=False)
        return hvp_flat
    
    A = LinearOperator((P, P), matvec=matvec, dtype=np.float64)
    eigvals, eigvecs = eigsh(A, k=1, which="LA", tol=config.eigsh_tol, maxiter=config.eigsh_maxiter)
    top = float(eigvals[0])
    w_top = torch.from_numpy(eigvecs[:, 0]).to(device=params[0].device, dtype=params[0].dtype)
    
    if tracker is not None:
        tracker.add(top, counter.total_equiv(), 0.0)
    
    return top, w_top


def vrpca_top_hessian_optimized(
    hvp_ops: HvpOps,
    counter: HvpCounter,
    config: AlgorithmConfig,
    budget: Optional[float] = None,
    seed: int = 0,
    tracker: Optional[ConvergenceTracker] = None,
    ref_eigenvalue: Optional[float] = None,
    progress: bool = False
) -> torch.Tensor:
    """Optimized VR-PCA with reduced overhead"""
    params = hvp_ops.params
    P = hvp_ops.num_params()
    device = params[0].device
    dtype = params[0].dtype
    rng = np.random.RandomState(seed)
    
    w_tilde = normalize_vec(torch.randn(P, device=device, dtype=dtype))
    n = hvp_ops.X.shape[0]
    
    # Optimized parameters
    if budget is not None and budget > 0:
        # Calculate optimal m given budget
        # Each epoch costs: 1 full + 2*m*batch_size/n equivalent HVPs
        cost_per_inner = 2.0 * config.batch_size / n
        epochs = min(config.vr_epochs_max, max(1, int(np.sqrt(budget))))
        m_per_epoch = max(1, int((budget / epochs - 1.0) / cost_per_inner))
        m = min(m_per_epoch, n)  # Cap at n
    else:
        m = int(config.vr_m_factor * n)
        epochs = config.vr_epochs_max
    
    # More aggressive step size for VR-PCA
    if config.vr_eta_mode == "adaptive":
        eta = hvp_ops.get_adaptive_stepsize()
    else:
        eta = config.vr_eta_fixed
    
    if config.verbose and progress:
        print(f"    VR-PCA: epochs={epochs}, m={m}, eta={eta:.4f}")
    
    start_time = time.perf_counter()
    
    for s in range(1, epochs + 1):
        # Compute full gradient at snapshot
        v_list_tilde = unflatten_like(w_tilde, params)
        Hw_tilde_list = hvp_full_counted(hvp_ops, counter, v_list_tilde)
        Hw_tilde_flat = flatten_params([h.detach() for h in Hw_tilde_list])
        
        w = w_tilde.clone()
        
        # Inner loop
        for t in range(1, m + 1):
            idx = rng.randint(0, n, size=min(config.batch_size, n))
            
            v_w = unflatten_like(w, params)
            v_tilde = unflatten_like(w_tilde, params)
            
            Hw_batch_w = hvp_minibatch_counted(hvp_ops, counter, v_w, idx)
            Hw_batch_tilde = hvp_minibatch_counted(hvp_ops, counter, v_tilde, idx)
            
            Hw_batch_w_flat = flatten_params([h.detach() for h in Hw_batch_w])
            Hw_batch_tilde_flat = flatten_params([h.detach() for h in Hw_batch_tilde])
            
            # Variance-reduced gradient
            update = Hw_batch_w_flat - Hw_batch_tilde_flat + Hw_tilde_flat
            w = normalize_vec(w + eta * update)
        
        w_tilde = w
        
        # Check convergence
        if tracker is not None:
            eigenval = rayleigh_quotient(hvp_ops, w_tilde)
            elapsed = time.perf_counter() - start_time
            tracker.add(eigenval, counter.total_equiv(), elapsed, ref_eigenvalue)
            
            if tracker.check_convergence(config.conv_tol):
                if progress:
                    print(f"    VR-PCA converged at epoch {s}/{epochs}")
                break
        
        if config.verbose and progress and s % max(1, epochs // 5) == 0:
            print(f"    VR-PCA: epoch {s}/{epochs}, HVPs: {counter.total_equiv():.2f}")
    
    return w_tilde


def oja_top_hessian(
    hvp_ops: HvpOps,
    counter: HvpCounter,
    config: AlgorithmConfig,
    budget: Optional[float] = None,
    seed: int = 0,
    tracker: Optional[ConvergenceTracker] = None,
    ref_eigenvalue: Optional[float] = None,
    progress: bool = False
) -> torch.Tensor:
    """Standard Oja algorithm (optional)"""
    params = hvp_ops.params
    P = hvp_ops.num_params()
    device = params[0].device
    dtype = params[0].dtype
    
    rng = np.random.RandomState(seed)
    w = torch.randn(P, device=device, dtype=dtype)
    w = normalize_vec(w)
    n = hvp_ops.X.shape[0]
    
    if budget is not None and budget > 0:
        max_iters = max(1, int(math.ceil(budget * n / config.batch_size)))
    else:
        max_iters = 10 * n
    
    eta0 = 0.01
    t0 = 10.0
    
    start_time = time.perf_counter()
    
    for t in range(1, max_iters + 1):
        idx = rng.randint(0, n, size=min(config.batch_size, n))
        v_list = unflatten_like(w, params)
        hvp_list = hvp_minibatch_counted(hvp_ops, counter, v_list, idx)
        hvp_flat = flatten_params([h.detach() for h in hvp_list])
        
        eta = eta0 / (t0 + t)
        w = normalize_vec(w + eta * hvp_flat)
        
        if tracker is not None and (t % max(1, max_iters // 20) == 0 or t == max_iters):
            eigenval = rayleigh_quotient(hvp_ops, w)
            elapsed = time.perf_counter() - start_time
            tracker.add(eigenval, counter.total_equiv(), elapsed, ref_eigenvalue)
    
    return w


# ---------------------------
# Analysis functions
# ---------------------------

def analyze_crossover_point(config: AlgorithmConfig) -> None:
    """Analyze theoretical crossover point between eigsh and VR-PCA"""
    print("\n" + "="*60)
    print("THEORETICAL ANALYSIS")
    print("="*60)
    
    print("\nVR-PCA theoretical complexity (from paper):")
    print("  O(n + 1/λ²) * log(1/ε)")
    print("\nEigsh complexity:")
    print("  O(n * log(1/ε) / λ)")
    print("\nExpected crossover when:")
    print("  1/λ² < n")
    print("  => λ > 1/√n")
    print("\nFor your experiments:")
    
    for N in [10, 100, 1000, 3000]:
        n = 3 * N  # dataset size
        lambda_threshold = 1 / np.sqrt(n)
        print(f"  N={N:4d}: crossover if λ > {lambda_threshold:.4f}")
    
    print("\nNote: Real performance depends on:")
    print("  - Constant factors in big-O")
    print("  - Implementation overhead")
    print("  - Actual eigenvalue gap λ")


def run_single_experiment(
    N: int, 
    config: AlgorithmConfig,
    run_oja: bool = False,
    data_mul: int = 3,
    d_in: int = 10,
    seed: int = 123,
    device: str = "cpu",
    dtype: str = "float32",
    progress: bool = True
) -> Dict[str, Any]:
    """Run experiment for a single N value"""
    
    torch.manual_seed(seed + N)
    np.random.seed(seed + N)
    
    dtype_torch = torch.float32 if dtype == "float32" else torch.float64
    loss_fn = nn.MSELoss()
    
    if progress:
        print(f"\n=== N={N} | params={(N+1)*d_in + N + 1} | dataset={data_mul * N} ===")
    
    model = make_model(N, d_in=d_in, device=device, dtype=dtype_torch)
    X, y = make_data(n_samples=data_mul * N, d_in=d_in, seed=seed + N,
                     device=device, dtype=dtype_torch)
    hvp_ops = HvpOps(model, loss_fn, X, y)
    
    result = {
        "N": N,
        "params": hvp_ops.num_params(),
        "dataset": int(X.shape[0]),
        "batch_size": config.batch_size
    }
    
    # Eigsh baseline
    eig_counter = HvpCounter()
    t0 = time.perf_counter()
    lam_ref, w_ref = top_hessian_eig_eigsh(hvp_ops, eig_counter, config)
    time_eigsh = time.perf_counter() - t0
    
    result["eigsh_time_s"] = time_eigsh
    result["eigsh_top_eig"] = lam_ref
    result["eigsh_equiv_full_hvps"] = eig_counter.total_equiv()
    
    if progress:
        print(f"  Eigsh:  time={time_eigsh:.3f}s, λ={lam_ref:.6g}, HVPs={eig_counter.total_equiv():.2f}")
    
    budget = eig_counter.total_equiv()
    
    # Optional Oja
    if run_oja:
        oja_counter = HvpCounter()
        t0 = time.perf_counter()
        w_oja = oja_top_hessian(
            hvp_ops, oja_counter, config, budget=budget, seed=seed,
            ref_eigenvalue=lam_ref, progress=progress
        )
        time_oja = time.perf_counter() - t0
        lam_oja = rayleigh_quotient(hvp_ops, w_oja)
        
        result["oja_time_s"] = time_oja
        result["oja_top_eig"] = lam_oja
        result["oja_equiv_full_hvps"] = oja_counter.total_equiv()
        result["rel_err_oja"] = abs(lam_oja - lam_ref) / (abs(lam_ref) + 1e-12)
        
        if progress:
            print(f"  Oja:    time={time_oja:.3f}s, λ≈{lam_oja:.6g}, err={result['rel_err_oja']:.3e}, HVPs={oja_counter.total_equiv():.2f}")
    
    # VR-PCA
    vr_counter = HvpCounter()
    vr_tracker = ConvergenceTracker() if config.track_convergence else None
    
    t0 = time.perf_counter()
    w_vr = vrpca_top_hessian_optimized(
        hvp_ops, vr_counter, config, budget=budget, seed=seed,
        tracker=vr_tracker, ref_eigenvalue=lam_ref, progress=progress
    )
    time_vr = time.perf_counter() - t0
    lam_vr = rayleigh_quotient(hvp_ops, w_vr)
    
    result["vrpca_time_s"] = time_vr
    result["vrpca_top_eig"] = lam_vr
    result["vrpca_equiv_full_hvps"] = vr_counter.total_equiv()
    result["rel_err_vrpca"] = abs(lam_vr - lam_ref) / (abs(lam_ref) + 1e-12)
    
    if progress:
        print(f"  VR-PCA: time={time_vr:.3f}s, λ≈{lam_vr:.6g}, err={result['rel_err_vrpca']:.3e}, HVPs={vr_counter.total_equiv():.2f}")
    
    # Store convergence data if tracked
    if config.track_convergence:
        result["vrpca_convergence"] = vr_tracker
    
    return result


def make_N_values(nmin: int, nmax: int, points: int) -> List[int]:
    """Generate N values with appropriate spacing"""
    if points <= 1:
        return [nmin]
    
    if nmax / nmin > 10:
        values = np.geomspace(nmin, nmax, points, dtype=int)
    else:
        values = np.linspace(nmin, nmax, points, dtype=int)
    
    values = sorted(set(values))
    if values[0] != nmin:
        values = [nmin] + values
    if values[-1] != nmax:
        values = values + [nmax]
    
    return values[:points] if len(values) > points else values


def plot_results(df: pd.DataFrame, out_prefix: str = "results", include_oja: bool = False):
    """Create plots from results DataFrame"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Runtime comparison
    ax = axes[0, 0]
    ax.loglog(df["N"], df["eigsh_time_s"], 'o-', label="Eigsh", linewidth=2, markersize=8)
    if include_oja and "oja_time_s" in df.columns:
        ax.loglog(df["N"], df["oja_time_s"], 's-', label="Oja", linewidth=2, markersize=6)
    ax.loglog(df["N"], df["vrpca_time_s"], '^-', label="VR-PCA", linewidth=2, markersize=8)
    ax.set_xlabel("Hidden size N")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Comparison (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    # Plot 2: HVP counts
    ax = axes[0, 1]
    ax.loglog(df["N"], df["eigsh_equiv_full_hvps"], 'o-', label="Eigsh", linewidth=2, markersize=8)
    if include_oja and "oja_equiv_full_hvps" in df.columns:
        ax.loglog(df["N"], df["oja_equiv_full_hvps"], 's-', label="Oja", linewidth=2, markersize=6)
    ax.loglog(df["N"], df["vrpca_equiv_full_hvps"], '^-', label="VR-PCA", linewidth=2, markersize=8)
    ax.set_xlabel("Hidden size N")
    ax.set_ylabel("Equivalent Full HVPs")
    ax.set_title("Computational Cost (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    # Plot 3: Relative error
    ax = axes[1, 0]
    if include_oja and "rel_err_oja" in df.columns:
        ax.semilogy(df["N"], df["rel_err_oja"], 's-', label="Oja", linewidth=2, markersize=6)
    ax.semilogy(df["N"], df["rel_err_vrpca"], '^-', label="VR-PCA", linewidth=2, markersize=8)
    ax.set_xlabel("Hidden size N")
    ax.set_ylabel("Relative Error")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    # Plot 4: Speedup
    ax = axes[1, 1]
    speedup_vr = df["eigsh_time_s"] / df["vrpca_time_s"]
    ax.plot(df["N"], speedup_vr, '^-', label="VR-PCA vs Eigsh", linewidth=2, markersize=8)
    if include_oja and "oja_time_s" in df.columns:
        speedup_oja = df["eigsh_time_s"] / df["oja_time_s"]
        ax.plot(df["N"], speedup_oja, 's-', label="Oja vs Eigsh", linewidth=2, markersize=6)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label="Break-even")
    ax.set_xlabel("Hidden size N")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("Speedup Relative to Eigsh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_prefix}_analysis.png")
    plt.close()


# ---------------------------
# Main CLI
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Hessian eigenvalue computation")
    
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # Run subcommand
    run_p = subparsers.add_parser("run", help="Run scaling experiments")
    run_p.add_argument("--nmin", type=int, default=10, help="Minimum hidden size")
    run_p.add_argument("--nmax", type=int, default=100, help="Maximum hidden size")
    run_p.add_argument("--points", type=int, default=10, help="Number of N values")
    run_p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    run_p.add_argument("--run_oja", action="store_true", help="Include Oja algorithm")
    run_p.add_argument("--out_csv", type=str, default="vrpca_results.csv")
    run_p.add_argument("--out_plots", type=str, default="vrpca")
    run_p.add_argument("--analyze", action="store_true", help="Show theoretical analysis")
    run_p.add_argument("--verbose", action="store_true", help="Verbose output")
    run_p.add_argument("--track_convergence", action="store_true", help="Track convergence")
    
    # Plot subcommand
    plot_p = subparsers.add_parser("plot", help="Plot existing results")
    plot_p.add_argument("--csv", type=str, required=True, help="Input CSV file")
    plot_p.add_argument("--out_prefix", type=str, default="results")
    
    args = parser.parse_args()
    
    if args.cmd == "plot":
        df = pd.read_csv(args.csv)
        include_oja = "oja_time_s" in df.columns
        plot_results(df, args.out_prefix, include_oja)
        return
    
    # Run experiments
    config = AlgorithmConfig(
        batch_size=args.batch_size,
        track_convergence=args.track_convergence,
        verbose=args.verbose,
        vr_m_factor=1.0,  # Reduced from 5.0
        vr_epochs_max=20
    )
    
    if args.analyze:
        analyze_crossover_point(config)
    
    N_values = make_N_values(args.nmin, args.nmax, args.points)
    results = []
    
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)
    
    for idx, N in enumerate(N_values):
        result = run_single_experiment(
            N=N,
            config=config,
            run_oja=args.run_oja,
            progress=True
        )
        
        result_for_csv = {k: v for k, v in result.items() if not k.endswith("_convergence")}
        results.append(result_for_csv)
        
        df_row = pd.DataFrame([result_for_csv])
        write_mode = 'w' if idx == 0 else 'a'
        df_row.to_csv(args.out_csv, mode=write_mode, header=write_mode == 'w', index=False)
    
    df = pd.DataFrame(results)
    print(f"\nResults saved to: {args.out_csv}")
    plot_results(df, args.out_plots, args.run_oja)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"VR-PCA vs Eigsh speedup: {(df['eigsh_time_s'] / df['vrpca_time_s']).mean():.2f}x average")
    print(f"VR-PCA accuracy: {df['rel_err_vrpca'].mean():.3e} average error")
    
    # Check if crossover happened
    speedups = df["eigsh_time_s"] / df["vrpca_time_s"]
    if (speedups > 1).any():
        crossover_N = df[speedups > 1]["N"].min()
        print(f"Crossover point: N={crossover_N} (VR-PCA faster than eigsh)")
    else:
        print(f"No crossover observed up to N={args.nmax}")
        print("Consider testing larger N values or different eigenvalue gaps")


if __name__ == "__main__":
    main()
