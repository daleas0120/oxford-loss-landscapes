# Examples

This directory contains example scripts demonstrating how to use the Oxford Loss Landscapes package.

## Available Examples

### Basic Usage
- `basic_usage.py` - Basic introduction to the package functionality
- `README.md` - This file

### Hessian Analysis  
- `hessian_analysis_guide.py` - Comprehensive guide to Hessian eigenvalue analysis concepts
- `hessian_eigenvalue_analysis.py` - Practical example of computing Hessian eigenvalues and eigenvectors
- `simple_hessian_analysis.py` - Simple demonstration of Hessian computation

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

Then run the examples:

```bash
# Basic package functionality
python examples/basic_usage.py

# Hessian eigenvalue analysis guide (concepts and theory)
python examples/hessian_analysis_guide.py

# Practical Hessian eigenvalue computation
python examples/hessian_eigenvalue_analysis.py
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

# Compute extreme eigenvalues and eigenvectors
max_eig, min_eig, max_eigvec, min_eigvec, iterations = min_max_hessian_eigs(
    model, inputs, targets, criterion
)

# Estimate trace (sum of eigenvalues)
trace = hessian_trace(model, criterion, inputs, targets)
```

## Example Categories

- **Basic Usage**: Introduction to core functionality
- **Hessian Analysis**: Second-order optimization analysis
- **Loss Landscapes**: Computing and visualizing loss landscapes  
- **Model Analysis**: Using metrics and model interface tools
- **Data Processing**: Working with model data and downloads
