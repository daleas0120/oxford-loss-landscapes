"""
Loss Landscapes: A library for visualizing and analyzing neural network loss landscapes
"""

__version__ = "0.3.0"
__author__ = "Loss Landscapes Contributors"
__email__ = "your.email@example.com"

# Main functionality
from .main import (
    point,
    linear_interpolation,
    random_line,
    planar_interpolation,
    random_plane,
)

# Model interface
from .model_interface.model_wrapper import ModelWrapper, GeneralModelWrapper

# Metrics
from .metrics import *

# Contrib modules
from .contrib import *

__all__ = [
    "point",
    "linear_interpolation", 
    "random_line",
    "planar_interpolation",
    "random_plane",
    "ModelWrapper",
    "GeneralModelWrapper",
]

