"""Basic tests for the oxford_loss_landscapes package."""

import pytest
import torch
from torch import nn


def test_package_import():
    """Test that the main package can be imported."""
    try:
        import oxford_loss_landscapes as oll
        assert hasattr(oll, 'point')
        assert hasattr(oll, 'ModelWrapper')
        assert hasattr(oll, 'download_zenodo_model')
    except ImportError as e:
        pytest.skip(f"Package not properly installed: {e}")


def test_model_wrapper():
    """Test basic ModelWrapper functionality."""
    try:
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        
        # Test wrapper creation
        wrapper = SimpleModelWrapper(model)
        assert wrapper is not None
        
        # Test forward pass
        test_input = torch.randn(10, 5)
        output = wrapper.forward(test_input)
        assert output.shape == (10, 1)
        
    except ImportError as e:
        pytest.skip(f"ModelWrapper not available: {e}")


def test_core_functions():
    """Test that core functions can be imported."""
    try:
        from oxford_loss_landscapes import (
            point, 
            linear_interpolation,
            random_line,
            random_plane
        )
        
        # Just test that functions exist
        assert callable(point)
        assert callable(linear_interpolation)
        assert callable(random_line)
        assert callable(random_plane)
        
    except ImportError as e:
        pytest.skip(f"Core functions not available: {e}")
