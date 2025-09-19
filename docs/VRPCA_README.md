# VR-PCA Hessian Analysis Guide

This guide explains the variance-reduced PCA (VR-PCA) tooling that now ships with
`oxford_loss_landscapes`, how it coexists with the classical `eigsh` Hessian
helpers, and how you can run the various scripts to obtain eigenvalue estimates,
plots, and scaling analyses.

## 1. Overview

- **Purpose**: Provide a scalable way to extract the dominant Hessian eigenpair
  using stochastic Hessian-vector products (HVPs) rather than dense eigendecomposition.
- **Location**: Core implementation lives in `src/oxford_loss_landscapes/hessian/vrpca/`.
- **Interface parity**: The public API mirrors the classical functions so examples and
  downstream code can switch between solvers with minimal edits.

## 2. Python API

Two primary entry points are exported from `oxford_loss_landscapes.hessian`:

```python
from oxford_loss_landscapes.hessian import (
    min_max_hessian_eigs,                 # classical eigsh-based solver
    top_hessian_eigenpair_vrpca,          # VR-PCA dominant eigenpair helper
    min_max_hessian_eigs_vrpca,           # classical-style tuple with VR-PCA max eigen
    VRPCAConfig,                          # configuration dataclass
)
```

### 2.1 `VRPCAConfig`

Configuration knobs, defined in `vrpca/config.py`, include batch size, epoch
count, convergence tolerance, inner-loop multiplier, and RNG seed. Typical usage:

```python
config = VRPCAConfig(batch_size=128, epochs=12, tol=1e-4)
result = top_hessian_eigenpair_vrpca(model, inputs, targets, criterion, config=config)
print(result.eigenvalue, result.converged, result.hvp_equivalent_calls)
```

### 2.2 Classical compatibility

`min_max_hessian_eigs_vrpca` returns a tuple shaped like the legacy
`min_max_hessian_eigs`. Minimum-eigenvalue entries are currently `None`, so older
code that expects `(max_eig, min_eig, max_vec, min_vec, iterations)` can upgrade
progressively.

## 3. Example scripts (`examples/`)

| Script | Description | How to run |
|--------|-------------|------------|
| `hessian_analysis_guide.py` | Conceptual walkthrough of classical Hessian analysis. | `python examples/hessian_analysis_guide.py` |
| `hessian_eigenvalue_analysis.py` | Practical classical analysis; now prints VR-PCA results when available. | `python examples/hessian_eigenvalue_analysis.py` |
| **`simple_hessian_vrpca_analysis.py`** | Minimal VR-PCA demo with optional classical comparison. | `python examples/simple_hessian_vrpca_analysis.py --compare` |
| **`vrpca_hessian_analysis_guide.py`** | New guide focusing exclusively on VR-PCA concepts, workflow, and interpretation. | `python examples/vrpca_hessian_analysis_guide.py` |

Each script injects `src/` onto `sys.path`, so running them from a source checkout
works once the correct conda environment is active:

```bash
source /home/alok/miniconda3/bin/activate Oxford_RSC  # or: conda activate Oxford_RSC
python examples/simple_hessian_vrpca_analysis.py
```

## 4. Research utilities (`vr_pca/`)

These scripts leverage the packaged VR-PCA implementation to reproduce scaling
studies and generate CSV outputs for plotting or further post-processing:

| Script | What it does | Example command |
|--------|--------------|-----------------|
| `hessian_scaling.py` | Compare VR-PCA against classical eigsh on a synthetic MLP. | `python vr_pca/hessian_scaling.py --solver both --width 64 --samples 512` |
| `scaling_analysis.py` | Sweep multiple widths/dataset sizes, saving results to CSV. | `python vr_pca/scaling_analysis.py --widths 64 128 --samples 512 1024 --output results.csv` |
| `combined_scaling_analysis.py` | Aggregate one or more CSVs into summary tables. | `python vr_pca/combined_scaling_analysis.py results.csv --output summary.csv` |

These scripts run entirely with the packaged API—no duplicated VR-PCA logic—so
updates to the library will automatically propagate.

## 5. Workflow examples

### 5.1 Obtain dominant eigenvalues

1. Train or load a model and dataset.
2. Call `top_hessian_eigenpair_vrpca(model, inputs, targets, loss_fn)`.
3. Inspect `result.eigenvalue` and `result.hvp_equivalent_calls` to gauge
   convergence cost.
4. (Optional) Call `min_max_hessian_eigs` for min/max pairs and compare.

### 5.2 Plot VR-PCA vs classical convergence

1. Run `python vr_pca/scaling_analysis.py --widths 64 128 --samples 1024 --repeats 3 --output scaling.csv`.
2. Use `combined_scaling_analysis.py` to summarise metrics: average eigenvalue,
   runtime, and HVP usage.
3. Plot columns from `scaling.csv` using your favourite notebook or tool.

### 5.3 Integrate into custom analysis

- Import `VRPCAConfig` and `top_hessian_eigenpair_vrpca` inside your existing
  loss-landscape workflows (`examples/simple_hessian_analysis.py`, dashboards, etc.).
- Use the classical solver for diagnostics (e.g., verifying the minimum
  eigenvalue) and the VR-PCA solver for scalable curvature estimates.

## 6. Testing

The regression suite `tests/test_hessian_vrpca.py` compares VR-PCA against the
classical baseline on a small model. Run inside the `Oxford_RSC` environment:

```bash
pytest tests/test_hessian_vrpca.py -q
```

SciPy is required (already listed as a project dependency). If SciPy or PyTorch
are missing, install via:

```bash
pip install -e .[dev]
```

## 7. Troubleshooting

- **Module not found**: Ensure the conda env is active or install the package in
  editable mode. The examples add `src/` to `sys.path`, so running them from the
  repository root works after activation.
- **Slow convergence**: Increase `VRPCAConfig.epochs`, tweak `batch_size`, or set
  `inner_loop_factor > 1` for more mini-batch steps per epoch.
- **Minimum eigenvalue needed**: Fall back to `min_max_hessian_eigs`—VR-PCA
  currently targets only the dominant eigenpair.

With these tools, you can explore Hessian curvature for larger models, benchmark
stochastic vs classical solvers, and integrate VR-PCA into existing loss landscape
pipelines.
