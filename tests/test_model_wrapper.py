import pytest
import torch
import torch.nn as nn
import numpy as np

def test_init_model_wrapper():
    try:
        from oxford_loss_landscapes.model_interface.model_wrapper import ModelWrapper
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.model_interface.model_wrapper import GeneralModelWrapper
    except ImportError as e:
        pytest.skip(f"ModelWrapper not available: {e}")     

def test_model_wrapper():
    """Test basic ModelWrapper functionality."""
    try:
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
    except ImportError as e:
        pytest.skip(f"ModelWrapper not available: {e}")
        
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

def test_simple_model_wrapper():
    """Test SimpleModelWrapper functionality."""
    try:
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
    except ImportError as e:
        pytest.skip(f"SimpleModelWrapper not available: {e}")
        
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
