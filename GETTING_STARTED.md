# Getting Started with Oxford Loss Landscapes

## Quick Installation

**One command installs everything:**

```bash
pip install -e .
```

This automatically installs:
- ✅ PyTorch (correct version for your Python)
- ✅ NumPy (correct version for your Python) 
- ✅ SciPy, pandas, tqdm, requests
- ✅ torchdiffeq and all other dependencies

**No manual PyTorch installation is needed!**

## Verify Installation

```python
import oxford_loss_landscapes as oll
import torch
print(f"✓ Oxford Loss Landscapes ready!")
print(f"✓ PyTorch {torch.__version__}")
```

## Quick Example

```python
import torch
import torch.nn as nn
import oxford_loss_landscapes as oll

# Create a model
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
criterion = nn.MSELoss()

# Generate data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Wrap model
from oxford_loss_landscapes.model_interface import SimpleModelWrapper
from oxford_loss_landscapes.metrics import Loss

model_wrapper = SimpleModelWrapper(model)
loss_metric = Loss(criterion, inputs, targets)

# Compute current loss
current_loss = oll.point(model_wrapper, loss_metric)
print(f"Current loss: {current_loss:.4f}")

# Create a 2D loss landscape
landscape = oll.random_plane(model_wrapper, loss_metric, distance=1.0, steps=25)
print(f"Loss landscape shape: {landscape.shape}")
```

## Python Version Support

- **Python 3.8-3.12**: Uses PyTorch >=1.8.0, NumPy <2.0
- **Python 3.13+**: Uses PyTorch >=2.5.0, NumPy >=2.0

Version selection is completely automatic - no configuration required!

## Next Steps

- [Examples directory](examples/) - Working examples for all features
- [Hessian Analysis Guide](HESSIAN_GUIDE.md) - Deep dive into second-order analysis
- [API Documentation](src/oxford_loss_landscapes/) - Full package reference
