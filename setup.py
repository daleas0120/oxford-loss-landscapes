#!/usr/bin/env python3
"""
Setup script for oxford-loss-landscapes package.

This setup.py is provided for compatibility and development purposes.
The package is primarily configured through pyproject.toml using modern
Python packaging standards.
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're running with Python 3.10+
if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or higher is required.")

# Get the directory containing this setup.py file
HERE = Path(__file__).parent.absolute()

# Read the README file for long description
README_PATH = HERE / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Loss Landscapes for Explainable AI - A library for visualizing and analyzing neural network loss landscapes"

# Core dependencies (matching pyproject.toml)
install_requires = [
    "torch>=1.8.0; python_version<'3.13'",
    "torch>=2.5.0; python_version>='3.13'",
    "torchdiffeq>=0.2.0",
    "numpy>=1.19.0,<2.0; python_version<'3.13'",
    "numpy>=2.0; python_version>='3.13'",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "tqdm>=4.60.0",
    "requests>=2.25.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "dash>=3.2.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "black>=21.0.0",
        "isort>=5.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
        "twine>=6.2.0",
        "build",
        "ipython",
        "ipykernel",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.17.0",
        "furo>=2023.2.11",
    ],
    "transformers": [
        "transformers>=4.20.0",
        "tokenizers>=0.13.0",
        "datasets>=2.0.0",
    ],
    "advanced": [
        "plotly>=5.0.0",
        "ipywidgets>=7.6.0",
        "jupyter>=1.0.0",
        "scikit-learn>=1.0.0",
        "torchvision>=0.13.0",
        "streamlit",
    ],
    "examples": [
        "torchvision",
        "datasets",
    ],
}

# Add 'all' extra that includes all optional dependencies
extras_require["all"] = list(set(
    dep for extra_deps in extras_require.values() for dep in extra_deps
))

# Entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "oxford-download-models=oxford_loss_landscapes.download_models:main",
    ],
}

# Package data
package_data = {
    "oxford_loss_landscapes": ["py.typed"],
}

setup(
    name="oxford-loss-landscapes",
    
    # Use setuptools_scm for version management
    use_scm_version={
        "write_to": "src/oxford_loss_landscapes/_version.py",
        "version_scheme": "release-branch-semver",
    },
    
    description="Loss Landscapes for Explainable AI - A library for visualizing and analyzing neural network loss landscapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Ashley S. Dale",
    author_email="trustworthy.ai.toolkit@gmail.com",
    maintainer="Ashley S. Dale",
    maintainer_email="trustworthy.ai.toolkit@gmail.com",
    
    url="https://github.com/daleas0120/oxford_rse_project4",
    project_urls={
        "Homepage": "https://github.com/daleas0120/oxford_rse_project4",
        "Repository": "https://github.com/daleas0120/oxford_rse_project4.git",
        "Bug Reports": "https://github.com/daleas0120/oxford_rse_project4/issues",
        "Documentation": "https://github.com/daleas0120/oxford_rse_project4#readme",
    },
    
    license="MIT",
    
    # Find packages in src/ directory
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Include package data
    package_data=package_data,
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points=entry_points,
    
    # Setuptools SCM for version management
    setup_requires=["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"],
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    
    keywords="machine learning neural networks loss landscapes explainable ai pytorch",
    
    # Zip safe
    zip_safe=False,
)
