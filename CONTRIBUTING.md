# Development and Contributing Guide

## Development Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/daleas0120/oxford_rse_project4.git
   cd oxford_rse_project4
   ```

2. **Install in development mode**:

   ```bash
   pip install -e .[dev]
   ```

3. **Run tests**:

   ```bash
   python -m pytest tests/ -v
   ```

## Package Structure

The package is organized as follows:

```
src/oxford_loss_landscapes/
├── __init__.py                 # Main package exports
├── main.py                     # Core loss landscape functions
├── download_models.py          # Model downloading utilities  
├── utils.py                    # Small analysis functions; move model between devices 
├── model_interface/            # Model wrapper interfaces
│   ├── model_wrapper.py        # Abstract and concrete wrappers
│   └── model_parameters.py     # Parameter manipulation utilities
├── metrics/                    # Loss and evaluation metrics
│   ├── metric.py               # Base metric classes
│   ├── sl_metrics.py           # Supervised learning metrics
│   └── rl_metrics.py           # Reinforcement learning metrics
├── contrib/                    # Additional algorithms
│   ├── connecting_paths.py     # Path-based analysis
│   └── trajectories.py         # Trajectory tracking
├── hessian/                    # Hessian computation tools
│   ├── hessian.py              # Main Hessian functions
│   ├── hessian_trace.py        # Trace computation
│   └── utilities.py            # Helper functions
└── dashboard/                  # Data analysis utilities
    ├── data_parser.py          # Data parsing functions
    └── model_parser.py         # Model analysis functions
```

## Contributing

1. **Code Style**: We use `black` for formatting and `isort` for import sorting:

   ```bash
   black src/ tests/
   isort src/ tests/
   ```

2. **Type Checking**: Use `mypy` for type checking:

   ```bash
   mypy src/
   ```

3. **Testing**: Add tests for new functionality in the `tests/` directory.

4. **Documentation**: Update docstrings and README.md as needed.

## Building and Distribution

1. **Build the package**:

   ```bash
   python -m build
   ```

2. **Check the distribution**:

   ```bash
   twine check dist/*
   ```

3. **Upload to PyPI** (maintainers only):

   ```bash
   twine upload dist/*
   ```

## Notes for Maintainers

- The package uses `setuptools_scm` for automatic versioning based on git tags
- Version information is automatically written to `src/oxford_loss_landscapes/_version.py`
- Dependencies are managed in `pyproject.toml`
- The package supports Python 3.10+

## Known Issues

1. **NumPy Compatibility**: Currently pinned to `numpy<2.0` due to PyTorch compatibility issues
2. **Missing Dependencies**: Some dashboard functions require additional modules (calibration, experiment) that need to be implemented
3. **Python 3.13 and PyTorch**: Sometimes fails on MacOS devices
