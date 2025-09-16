# Oxford Loss Landscapes

Loss Landscapes for EXplainable AI - A Python library for visualizing and analyzing neural network loss landscapes.

## Overview

This package provides tools for:

- Computing and visualizing loss landscapes of neural networks
- Analyzing Hessian properties of loss functions
- Downloading and managing models from research datasets
- Dashboard utilities for data analysis and visualization

## Installation

### Simple Installation (Recommended)

The package automatically installs all dependencies including PyTorch:

```bash
pip install -e .
```

This single command installs:

- PyTorch (with automatic version selection for your Python version)  
- All numerical computation dependencies (NumPy, SciPy, pandas)
- Visualization tools (matplotlib, seaborn)
- Package-specific dependencies (torchdiffeq, etc.)

### Development Installation

For development with testing and documentation tools:

```bash
pip install -e .[dev]
```

### Optional Dependencies

For transformer model support:

```bash
pip install -e ".[transformers]"

```

For advanced visualization and analysis tools:

```bash
pip install -e ".[advanced]"
```

This includes: plotly, ipywidgets, jupyter, scikit-learn, torchvision

### Installation with requirements.txt

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Python Version Compatibility

- **Python 3.8-3.12**: Uses PyTorch >=1.8.0 and NumPy <2.0
- **Python 3.13+**: Uses PyTorch >=2.5.0 and NumPy >=2.0 (automatic version selection)

## Quick Start

```python
import torch
import torch.nn as nn
import oxford_loss_landscapes as oll

# Create a simple model and loss function
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
criterion = nn.MSELoss()

# Generate some dummy data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Wrap the model
model_wrapper = oll.ModelWrapper(model, criterion, inputs, targets)

# Compute a random 2D loss landscape
landscape = oll.random_plane(model_wrapper, distance=1.0, steps=25)
print(f"Loss landscape shape: {landscape.shape}")

# Compute loss at current parameters
loss_value = oll.point(model_wrapper)
print(f"Current loss: {loss_value}")
```

## Features

### Core Loss Landscape Functions

- `point()`: Evaluate loss at current parameters
- `linear_interpolation()`: Loss along a line between two points

- `random_line()`: Loss along a random direction
- `planar_interpolation()`: Loss over a 2D plane between three points  
- `random_plane()`: Loss over a random 2D plane

### Model Interface

- `ModelWrapper`: Interface for PyTorch models
- `SimpleModelWrapper`: Interface for simple models

### Utilities

- Model downloading from Zenodo datasets
- Hessian computation tools
- Dashboard data parsing utilities

## Command Line Tools

Download models from research datasets:

```bash
oxford-download-models --output models.zip --extract-dir ./models
```

## Project Structure

```
oxford_rse_project4/
├── src/oxford_loss_landscapes/     # Main package
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Core loss landscape functions
│   ├── download_models.py          # Model downloading utilities
│   ├── model_interface/            # Model wrapper interfaces
│   ├── metrics/                    # Loss and evaluation metrics
│   ├── contrib/                    # Additional algorithms
│   ├── hessian/                    # Hessian computation tools
│   └── dashboard/                  # Data analysis utilities
├── scripts/                        # Utility scripts
├── models/                         # Downloaded model storage
└── pyproject.toml                  # Modern Python packaging config
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable

5. Submit a pull request

## Requirements

- Python >=3.8 (Python 3.12 recommended for best compatibility)
- pip package manager

**All other dependencies are automatically installed**, including:

- PyTorch (>=1.8.0 for Python <3.13, >=2.5.0 for Python >=3.13)
- NumPy (version automatically selected based on Python version)
- SciPy, pandas, tqdm, requests, torchdiffeq
- Visualization tools: matplotlib, seaborn

No manual installation of dependencies required!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this library in your research, please consider citing:

```bibtex
@software{oxford_loss_landscapes,
  title = {Oxford Loss Landscapes: A library for visualizing and analyzing neural network loss landscapes},
  author = {Oxford RSE Project Contributors},
  url = {https://github.com/daleas0120/oxford_rse_project4},
  year = {2025}
}
```
