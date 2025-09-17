# Hessian Eigenvalue Analysis with Oxford Loss Landscapes

## Overview

This document provides comprehensive guidance on calculating eigenvalues and eigenvectors of the Hessian matrix for neural network models using the `oxford_loss_landscapes.hessian` package.

## Quick Start

### Installation Requirements

Install the package (all dependencies included automatically):

```bash
pip install -e .
```

This installs PyTorch, NumPy, SciPy, pandas, and all other required dependencies with automatic version selection for your Python version.

### Basic Usage

```python
import torch
import torch.nn as nn
from oxford_loss_landscapes.model_interface.model_wrapper import ModelWrapper
from oxford_loss_landscapes.hessian.hessian import min_max_hessian_eigs
from oxford_loss_landscapes.hessian.hessian_trace import hessian_trace

# 1. Create your model
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 2. Prepare data
X = torch.randn(100, 4)
y = torch.randn(100, 1)

# 3. Define loss function
def loss_fn(model_output):
    return nn.MSELoss()(model_output, y)

# 4. Wrap your model
model_wrapper = ModelWrapper(model, loss_fn, X)

# 5. Compute Hessian eigenvalues
min_eig, max_eig, min_eigvec, max_eigvec = min_max_hessian_eigs(
    model_wrapper, 
    use_cuda=torch.cuda.is_available()
)

# 6. Estimate Hessian trace
trace = hessian_trace(model_wrapper, num_random_vectors=50)

print(f"Minimum eigenvalue: {min_eig}")
print(f"Maximum eigenvalue: {max_eig}")
print(f"Condition number: {max_eig / min_eig if min_eig > 0 else 'inf'}")
print(f"Hessian trace estimate: {trace}")
```

## Available Functions

### `min_max_hessian_eigs(model_wrapper, loss_fn=None, use_cuda=False, max_iter=100, tol=1e-3)`

**Purpose**: Compute the minimum and maximum eigenvalues of the Hessian matrix

**Parameters**:

- `model_wrapper`: Wrapped model with loss function and data
- `loss_fn`: Optional loss function (uses wrapper's if None)
- `use_cuda`: Whether to use GPU acceleration
- `max_iter`: Maximum iterations for eigenvalue computation
- `tol`: Convergence tolerance

**Returns**: `(min_eigenvalue, max_eigenvalue, min_eigenvector, max_eigenvector)`

### `hessian_trace(model_wrapper, loss_fn=None, num_random_vectors=100, use_cuda=False)`

**Purpose**: Estimate the trace of the Hessian matrix using Hutchinson's method

**Parameters**:

- `model_wrapper`: Wrapped model with loss function and data
- `loss_fn`: Optional loss function (uses wrapper's if None)
- `num_random_vectors`: Number of random vectors for trace estimation
- `use_cuda`: Whether to use GPU acceleration

**Returns**: `estimated_trace` (float)

## Interpreting Results

### Eigenvalues

**Minimum Eigenvalue**:

- **Negative**: Indicates saddle point or local maximum
- **Near zero**: Very flat region, potential optimization challenges
- **Positive**: Local minimum with convex behavior

**Maximum Eigenvalue**:

- **Large values**: Steep loss changes in some directions
- **Affects**: Optimization stability and learning rate selection

### Condition Number (`max_eig / min_eig`)

- **Measures**: Optimization difficulty
- **Higher values**: More challenging optimization
- **Values > 1000**: Often suggest numerical instability

### Eigenvectors

- **Show**: Directions of extreme curvature
- **Usage**: Understanding parameter contributions
- **Applications**: Parameter space analysis and optimization

### Trace

- **Represents**: Sum of all eigenvalues
- **Relates to**: Overall "sharpness" of loss landscape
- **Usage**: Comparing different models or training states

## Practical Applications

### 1. Learning Rate Selection

- Use smaller learning rates for larger maximum eigenvalues
- Consider the condition number for stability

### 2. Optimization Method Choice

- High condition numbers may benefit from second-order methods
- Low condition numbers work well with first-order methods

### 3. Model Comparison

- Compare sharpness across different architectures
- Evaluate training state changes

### 4. Training Dynamics

- Monitor how the loss landscape changes during training
- Identify convergence characteristics

## Example Workflows

### Workflow 1: Model Analysis

```python
# Train your model first
model.train()
for epoch in range(num_epochs):
    # ... training loop ...

# Analyze the trained model
model.eval()
model_wrapper = ModelWrapper(model, loss_fn, X)

# Compute Hessian properties
min_eig, max_eig, _, _ = min_max_hessian_eigs(model_wrapper)
trace = hessian_trace(model_wrapper)

# Interpret results
condition_number = max_eig / min_eig if min_eig > 0 else float('inf')
print(f"Model sharpness analysis:")
print(f"  Condition number: {condition_number:.2f}")
print(f"  Loss landscape sharpness: {trace:.4f}")
```

### Workflow 2: Training Monitoring

```python
# Monitor during training
eigenvalue_history = []

for epoch in range(num_epochs):
    # ... training step ...
    
    if epoch % 10 == 0:  # Check every 10 epochs
        model_wrapper = ModelWrapper(model, loss_fn, X)
        min_eig, max_eig, _, _ = min_max_hessian_eigs(model_wrapper)
        eigenvalue_history.append((epoch, min_eig, max_eig))

# Analyze training dynamics
for epoch, min_eig, max_eig in eigenvalue_history:
    condition = max_eig / min_eig if min_eig > 0 else float('inf')
    print(f"Epoch {epoch}: Condition number = {condition:.2f}")
```

## Troubleshooting

### Common Issues

1. **NumPy Compatibility**
   - **Automatic**: The package automatically handles NumPy version compatibility
   - **Python <3.13**: Uses NumPy <2.0 automatically
   - **Python >=3.13**: Uses NumPy >=2.0 automatically

2. **Memory Issues with Large Models**
   - **Solution**: Use smaller `num_random_vectors` for trace estimation
   - **Solution**: Reduce `max_iter` for eigenvalue computation

3. **Convergence Issues**
   - **Solution**: Increase `tol` parameter
   - **Solution**: Ensure model is properly trained

4. **CUDA Errors**
   - **Solution**: Set `use_cuda=False` if GPU issues occur
   - **Solution**: Ensure PyTorch CUDA installation is correct

### Performance Tips

1. **For Large Models**: Start with small `num_random_vectors` (10-20)
2. **For Precision**: Increase `num_random_vectors` (100-200)
3. **For Speed**: Use GPU when available (`use_cuda=True`)
4. **For Stability**: Use lower tolerance (`tol=1e-4` or smaller)

## Advanced Usage

### Custom Loss Functions

```python
def custom_loss_fn(model_output):
    # Your custom loss computation
    mse_loss = nn.MSELoss()(model_output, y)
    regularization = 0.01 * sum(p.norm() for p in model.parameters())
    return mse_loss + regularization

model_wrapper = ModelWrapper(model, custom_loss_fn, X)
```

### Batch Processing

```python
# For large datasets, use batched analysis
batch_size = 32
results = []

for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    def batch_loss_fn(model_output):
        return nn.MSELoss()(model_output, y_batch)
    
    wrapper = ModelWrapper(model, batch_loss_fn, X_batch)
    min_eig, max_eig, _, _ = min_max_hessian_eigs(wrapper)
    results.append((min_eig, max_eig))

# Analyze batch results
avg_min_eig = sum(r[0] for r in results) / len(results)
avg_max_eig = sum(r[1] for r in results) / len(results)
```

## Current Limitations

1. **NumPy 2.x Compatibility**: Currently requires NumPy < 2.0 due to PyTorch compatibility
2. **Memory Usage**: Large models may require significant memory for eigenvalue computation
3. **Computation Time**: Hessian analysis can be computationally expensive for large models

## Getting Help

If you encounter issues:

1. Check that all dependencies are properly installed
2. Ensure NumPy version compatibility
3. Try with a smaller model first to verify functionality
4. Review the example files in the `examples/` directory

## Examples Available

- `hessian_usage_demo.py`: Basic usage demonstration
- `hessian_analysis_guide.py`: Conceptual guide and interpretation
- `hessian_eigenvalue_analysis.py`: Comprehensive practical example
- `simple_hessian_analysis.py`: Minimal working example

Run any example with:

```bash
python examples/example_name.py
```

## References

The Hessian analysis functionality is based on established methods in optimization theory and neural network analysis, using efficient implementations for eigenvalue computation and trace estimation.
