# Installation Guide

## Requirements

- Python 3.8+ (recommended: Python 3.12 for best PyTorch compatibility)
- pip or conda package manager

## Installation

### Option 1: Direct pip installation (Recommended)

For Python 3.12 and below:
```bash
pip install -e .
```

This will automatically install all dependencies including:
- PyTorch (>=1.8.0)
- NumPy (<2.0 for Python <3.13, >=2.0 for Python >=3.13)
- pandas, scipy, tqdm, requests
- torchdiffeq

### Option 2: Using conda environment

1. Create a conda environment with Python 3.12:
```bash
conda create -n oxford_loss python=3.12
conda activate oxford_loss
```

2. Install the package:
```bash
pip install -e .
```

## Verification

Test the installation:
```python
import oxford_loss_landscapes
import torch
print(f"Installation successful! PyTorch version: {torch.__version__}")
```

## Notes

- Python 3.13 support: The package automatically uses NumPy >=2.0 and PyTorch >=2.5.0 for Python 3.13+
- Python 3.12 and below: Uses NumPy <2.0 and PyTorch >=1.8.0 for broader compatibility
- All PyTorch dependencies are now included in the main package dependencies and install automatically
