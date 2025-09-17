# VR-PCA Hessian Experiments

This folder contains experimental tooling for benchmarking a variance-reduced
principal component analysis (VR-PCA) estimator of the top Hessian eigenvalue
against the Lanczos/eigsh baseline used throughout the
`oxford_loss_landscapes` package. The code is derived from the algorithm
presented in Shamir et al.'s *Stochastic PCA* notes (`Shamir_Stochastic_PCA_1-5.pdf`) and
reuses the same Hessian–vector product (HVP) machinery that powers
`src/oxford_loss_landscapes/hessian/hessian.py` and
`src/oxford_loss_landscapes/hessian/hessian_trace.py`.

The scripts here generate synthetic regression problems, run both solvers with
matched HVP budgets, and report runtime, accuracy, and scaling trends. They are
intended for research and validation rather than as part of the published API,
but they illustrate how the VR-PCA routine could be integrated into the main
library for faster large-scale loss landscape analysis.

## Files

- `hessian_scaling.py`: CLI entry point that constructs toy models, runs eigsh,
  the proposed VR-PCA routine, and (optionally) Oja's algorithm, then writes a
  metrics CSV and companion plots.
- `scaling_analysis.py`: Post-processing tool that fits power-law scaling curves
  and visualises the runtime versus accuracy trade-off using the CSV emitted by
  `hessian_scaling.py`.
- `combined_scaling_analysis.py`: Variant of the scaling script focused on
  producing a single publication-style plot plus textual interpretation of the
  fitted exponents.
- `Shamir_Stochastic_PCA_1-5.pdf`: Notes describing the VR-PCA method this
  implementation follows.

## Background: VR-PCA vs eigsh

`src/oxford_loss_landscapes/hessian/hessian.py` wraps SciPy's `eigsh`, which in
turn applies a Lanczos iteration requiring repeated full-batch HVPs. While the
routine is robust, its cost grows with the number of parameters and the desired
precision. The VR-PCA algorithm side-steps this by combining:

1. Periodic **snapshot** computations of the full HVP at the current iterate.
2. **Mini-batch corrections** that approximate the HVP on small random subsets
   of the data.
3. A **variance-reduced update** that adds the mini-batch difference to the
   snapshot, yielding an unbiased HVP estimate with substantially lower noise.
4. **Normalization** after every step to keep the iterate on the unit sphere.

When the leading Hessian eigenvalue has a moderate gap (λ) and the dataset is
large, the paper shows VR-PCA converges to the top eigenvector in
`O(n + 1/λ²) log(1/ε)` HVPs—favourable compared to the `O(n · log(1/ε) / λ)`
complexity of Lanczos. The implementation in `hessian_scaling.py` adds:

- An adaptive stepsize based on a power-iteration estimate of the Hessian
  spectral norm.
- A dynamic inner loop length (`m`) chosen to respect the Lanczos HVP budget.
- Optional tracking of convergence curves (eigenvalue estimate, effective HVPs,
  runtime) for diagnostics.

Because every VR-PCA update still calls `HvpOps.hvp_for_batch`, integrating the
routine into the package would only require exposing a public API that shares a
`LinearOperator` interface with the existing eigsh wrapper.

## Installation

All scripts rely on the editable install of the project:

```bash
python -m pip install -e .[dev]
```

Key runtime dependencies include `torch`, `numpy`, `scipy`, `pandas`,
`matplotlib`, and `scikit-learn`. GPU execution is not required—the default
configuration runs entirely on CPU with synthetic data.

## Running the scaling experiments

The main entry point accepts a `run` subcommand. Example:

```bash
python vr_pca/hessian_scaling.py run \
    --nmin 16 --nmax 512 --points 8 \
    --batch_size 64 \
    --run_oja \
    --track_convergence \
    --out_csv vrpca_results.csv \
    --out_plots vrpca
```

This command:

- Sweeps hidden-layer widths `N` between 16 and 512 (log-spaced across eight
  points) while keeping the input dimension at 10 and dataset size at `3 × N`.
- Measures runtime (`*_time_s`), equivalent full-batch HVP usage
  (`*_equiv_full_hvps`), and the Rayleigh-quotient estimate of the top
  eigenvalue for eigsh, VR-PCA, and Oja (if requested).
- Writes `vrpca_results.csv` and produces `vrpca_analysis.png`, a four-panel
  plot comparing runtime, HVP counts, relative error, and speed-up.

Pass `--analyze` to print the theoretical crossover criterion (λ ≳ 1/√n) from
Shamir et al., and `--verbose` for periodic progress logs. The CLI seeds both
NumPy and PyTorch so repeated runs are reproducible.

### Output columns

`hessian_scaling.py` stores one row per model size with the following notable
fields:

- `N`: hidden dimension of the synthetic network.
- `params` / `dataset`: number of trainable parameters and samples.
- `eigsh_time_s`, `vrpca_time_s`, `oja_time_s`: wall-clock runtime in seconds.
- `eigsh_equiv_full_hvps`, `vrpca_equiv_full_hvps`: number of full-batch HVPs,
  counting mini-batch calls proportionally.
- `eigsh_top_eig`, `vrpca_top_eig`: estimated leading eigenvalues.
- `rel_err_vrpca`: relative error against the eigsh baseline.

If `--track_convergence` is enabled, the returned Python objects also contain a
`ConvergenceTracker` with per-epoch trajectories that can be inspected inside a
notebook for further analysis.

## Post-processing and visualisation

Given the CSV emitted above, the analysis scripts add publication-quality
visuals and fitted scaling laws.

```bash
python vr_pca/scaling_analysis.py vrpca_results.csv --plot-type all \
    --scaling-plot scaling.png \
    --error-plot runtime_vs_error.png \
    --combined-plot combined.png
```

- `plot_combined_scaling` fits power-law and offset models to the eigsh and
  VR-PCA runtimes (via non-linear least squares) and overlays their predictions
  on a log–log plot. The fitted equations and R² scores are stamped directly on
  the figure.
- `plot_runtime_vs_error` renders a parametric trade-off plot that joins the
  eigsh (zero-error baseline) and VR-PCA points for each `N` to highlight where
  the stochastic method gains speed at the cost of slight error.
- `create_combined_figure` places both views side-by-side with a shared colour
  scale over `N`.

For a more compact report, you can instead run:

```bash
python vr_pca/combined_scaling_analysis.py vrpca_results.csv \
    --output vrpca_scaling.png --verbose
```

which emits the scaling plot and, with `--verbose`, prints the fitted exponents
as a textual summary (e.g. "VR-PCA scaling: N^1.35").

## Relation to the main package

- **Shared primitives**: Both the experiments and the production code rely on
  Hessian–vector products computed with autograd. `HvpOps` wraps this logic and
  could be reused inside `oxford_loss_landscapes.hessian` for a VR-PCA backend.
- **Benchmarking**: The CSV and plots provide empirical guidance on when the
  variance-reduced method outperforms `eigsh`, informing whether a future
  `LossLandscapeAnalyzer` should expose a "stochastic" mode.
- **Extensibility**: The VR-PCA loop expects nothing about the model beyond the
  ability to evaluate HVPs. Swapping the synthetic MLP for a checkpoint loaded
  via `oxford_loss_landscapes.model_zoo` would allow apples-to-apples
  comparisons on real architectures.

## Example workflow

1. Install the project in editable mode with `[dev]` extras.
2. Run the scaling sweep to generate a CSV and raw plots (`hessian_scaling.py`).
3. Use `scaling_analysis.py` to produce publication-ready figures or
   `combined_scaling_analysis.py` for a quick read-out of fitted exponents.
4. Inspect the CSV to locate the point where VR-PCA becomes faster than eigsh
   while meeting your accuracy target.
5. Prototype integrating `vrpca_top_hessian_optimized` into the main library
   using these heuristics for step size and budget selection.

## Troubleshooting

- **SciPy or scikit-learn import errors**: ensure `pip install -e .[dev]` ran in
  a clean virtual environment; both packages are optional dependencies outside
  the experiments.
- **Long runtimes for large `N`**: increase `--batch_size` or reduce
  `--vr_epochs_max` to cap the VR-PCA budget, or skip the `--run_oja` baseline.
- **Plots not showing up**: the scripts always save figures to disk; open the
  generated PNGs manually or add `plt.show()` in a notebook context.

These tools complement the existing Hessian utilities by stress-testing a
variance-reduced alternative and documenting when it could be worth promoting to
first-class support inside `oxford_loss_landscapes`.
