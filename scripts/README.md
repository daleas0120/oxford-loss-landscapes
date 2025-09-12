# Scripts Directory

This directory contains utility scripts for the Oxford Loss Landscapes project.

## Files

- `get_gnn_model.sh` - Shell script for downloading GNN models
- `get_nn_model.sh` - Shell script for downloading neural network models  
- `get_transformer_model.py` - Python script for downloading transformer models

## Usage

These scripts are legacy utilities. For new model downloading functionality, use the main package:

```bash
oxford-download-models --help
```

Or in Python:

```python
from oxford_loss_landscapes import download_model
download_model()
```
