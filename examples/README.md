# Examples

This directory contains example scripts demonstrating how to use the Oxford Loss Landscapes package.

## Available Examples

### Basic Usage

- `basic_usage.py` - Basic introduction to the package functionality
- `transformer_usage.py` - Basic introduction to using the Hugging Face Transformer Wrapper
- `README.md` - This file

### Hessian Analysis

- `hessian_analysis_guide.py` – Conceptual walk-through plus CLI snippets (classical + VR-PCA tips)
- `simple_hessian_analysis.py` – Minimal classical `eigsh` example on a toy MLP
- `simple_hessian_vrpca_analysis.py` – Minimal VR-PCA pipeline, includes drop-in comparison vs classical
- `vrpca_hessian_analysis_guide.py` – Narrative guide to VR-PCA, error tracking, and HVP budgets
- `full_hessian_analysis_comparison.py` – Side-by-side timings and accuracy for both solvers
- `comprehensive_hessian_analysis.py` – Larger end-to-end pipeline highlighting VR-PCA as a drop-in backend
- `hessian_eigenvalue_analysis.py` – Practical classical analysis with optional VR-PCA output

### Planned Examples

- Loss landscape visualization examples
- Model comparison examples  
- Advanced metrics usage examples

## Running Examples

Install the package (includes all dependencies automatically):

```bash
pip install -e .
```

This single command installs:

- PyTorch (version automatically selected for your Python version)  
- All numerical computation dependencies (NumPy, SciPy, pandas)
- Package-specific dependencies (torchdiffeq, etc.)

> Note: To use the transformers and advanced visualization, need to install additional dependencies
>
> ```bash
> pip install -e ".[transformers]"
> ```
>
> and
>
>```bash
> pip install -e ".[advanced]"
> ```

Then run the examples:

```bash
# Basic package functionality
python examples/basic_usage.py

# Hessian eigenvalue analysis guide (concepts and theory)
python examples/hessian_analysis_guide.py

# Practical Hessian eigenvalue computation (classical + VR-PCA)
python examples/hessian_eigenvalue_analysis.py --compare

# Minimal VR-PCA example with classical comparison
python examples/simple_hessian_vrpca_analysis.py --compare
```

## Hessian Analysis Examples

The Hessian analysis examples demonstrate how to:

- Compute maximum and minimum eigenvalues of the Hessian matrix
- Extract and analyze eigenvectors  
- Interpret eigenvalue signs (local minimum vs saddle point)
- Estimate the Hessian trace using Hutchinson's method
- Understand conditioning and curvature properties of the loss landscape

### Key Functions Demonstrated

```python
from oxford_loss_landscapes.hessian import min_max_hessian_eigs, hessian_trace
from oxford_loss_landscapes.hessian.vrpca import (
    VRPCAConfig,
    min_hessian_eigenpair_vrpca,
    top_hessian_eigenpair_vrpca,
)

# Compute extreme eigenvalues and eigenvectors
max_eig, min_eig, max_eigvec, min_eigvec, iterations = min_max_hessian_eigs(
    model, inputs, targets, criterion
)

# Switch to VR-PCA (drop-in replacement)
config = VRPCAConfig(batch_size=128, epochs=12)
max_eig_vr, min_eig_vr, *_ = min_max_hessian_eigs(
    model,
    inputs,
    targets,
    criterion,
    backend="vrpca",
    vrpca_config=config,
    compute_min=True,
)

# Direct eigenpair helpers when only one extreme is required
max_result = top_hessian_eigenpair_vrpca(model, inputs, targets, criterion, config=config)
min_result = min_hessian_eigenpair_vrpca(model, inputs, targets, criterion, config=config)

# Estimate trace (sum of eigenvalues)
trace = hessian_trace(model, criterion, inputs, targets)
```
