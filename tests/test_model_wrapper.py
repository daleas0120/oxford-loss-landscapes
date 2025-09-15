"""Tests for the model loading and wrapping API."""

import pytest
import torch
import torch.nn as nn
import numpy as np

def test_init_model_wrapper():
    """Test that ModelWrapper can be imported and initialized."""
    try:
        from oxford_loss_landscapes.model_interface.model_wrapper import ModelWrapper
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
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
        nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 5),
            nn.ReLU(),
        ),      
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

    # Test get_modules
    # modules = wrapper.get_modules()
    # print(modules)
    # assert len(modules) == 4  # 3 Linear + 1 ReLU

    # Test get_module_params
    # params = wrapper.get_module_params(0)
    # assert len(params) == 2  # weight and bias of first Linear layer

    # Test train_mode and eval_mode
    # wrapper.train_mode()
    # for module in wrapper.get_modules():
    #     assert module.training is True
    # wrapper.eval_mode()
    # for module in wrapper.get_modules():
    #     assert module.training is False
    
    # Test requires_grad_
    # wrapper.requires_grad_(False)
    # for module in wrapper.get_modules():
    #     for p in module.parameters():
    #         assert p.requires_grad is False
    # wrapper.requires_grad_(True)
    # for module in wrapper.get_modules():
    #     for p in module.parameters():
    #         assert p.requires_grad is True

    # Test zero_grad
    # First, create some gradients
    # test_input = torch.randn(10, 5)
    # output = wrapper.forward(test_input)
    # output.sum().backward()
    # # Now zero them
    # wrapper.zero_grad()
    # for module in wrapper.get_modules():
    #     for p in module.parameters():
    #         if p.grad is not None:
    #             assert torch.all(p.grad == 0)
    #         else:
    #             assert p.grad is None
    
    # # Test parameters and named_parameters
    # params_list = list(wrapper.parameters())
    # named_params_list = list(wrapper.named_parameters())
    # assert len(params_list) == sum(1 for _ in model.parameters())
    # assert len(named_params_list) == sum(1 for _ in model.named_parameters())


