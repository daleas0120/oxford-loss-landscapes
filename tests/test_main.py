import pytest
import torch
from torch import nn
import numpy as np

def test_package_import():
    """Test that the main package can be imported."""
    try:
        import oxford_loss_landscapes as oll
        assert hasattr(oll, 'point')
        assert hasattr(oll, 'linear_interpolation')
        assert hasattr(oll, 'random_line')
        assert hasattr(oll, 'random_plane')
        assert hasattr(oll, 'planar_interpolation')
    except ImportError as e:
        pytest.skip(f"Package not properly installed: {e}")

def test_random_plane():
    """Test the random_plane function."""
    try:
        from oxford_loss_landscapes import random_plane
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics import Loss
    except ImportError:
        pytest.skip("Core functions not available; skipping related tests.")
    
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)

    model_wrapper = SimpleModelWrapper(model)
    criterion = nn.MSELoss()
    loss_metric = Loss(criterion, X, y)

    # Generate a random plane
    plane = random_plane(model_wrapper, loss_metric, distance=1, steps=5)

    # Check that the plane is a list of lists with correct dimensions
    assert isinstance(plane, np.ndarray)
    assert len(plane) == 5
    for point in plane:
        assert isinstance(point, np.ndarray)

def test_planar_interpolation():
    """Test the planar_interpolation function."""
    try:
        from oxford_loss_landscapes import planar_interpolation
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics import Loss
        from oxford_loss_landscapes.hessian.utilities import copy_wts_into_model
    except ImportError:
        pytest.skip("Core functions not available; skipping related tests.")
    
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)

    model_wrapper = SimpleModelWrapper(model)
    criterion = nn.MSELoss()
    loss_metric = Loss(criterion, X, y)
    
    # Create two random directions
    direction1 = [torch.randn_like(p) for p in model.parameters()]
    direction2 = [torch.randn_like(p) for p in model.parameters()]
    
    # Wrap directions to have the same shape as model parameters
    model_1 = SimpleModelWrapper(copy_wts_into_model(direction1, model))
    model_2 = SimpleModelWrapper(copy_wts_into_model(direction2, model))

    # Generate a planar interpolation
    plane = planar_interpolation(model_wrapper, model_1, model_2, loss_metric, distance=1, steps=5)

    # Check that the plane is a list of lists with correct dimensions
    assert isinstance(plane, np.ndarray)
    assert len(plane) == 5
    for point in plane:
        assert isinstance(point, np.ndarray)

def test_random_line():
    """Test the random_line function."""
    try:
        from oxford_loss_landscapes import random_line
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics import Loss
        from oxford_loss_landscapes.hessian.utilities import copy_wts_into_model
    except ImportError:
        pytest.skip("Core functions not available; skipping related tests.")
    
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)

    model_wrapper = SimpleModelWrapper(model)
    criterion = nn.MSELoss()
    loss_metric = Loss(criterion, X, y)


    # Generate a random line
    line = random_line(model_wrapper, loss_metric, distance=1, steps=5)

    # Check that the line is a list with correct dimensions
    assert isinstance(line, np.ndarray)
    assert len(line) == 5
    for point in line:
        assert isinstance(point, float)

def test_linear_interpolation():
    """Test the linear_interpolation function."""
    try:
        from oxford_loss_landscapes import linear_interpolation
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics import Loss
        from oxford_loss_landscapes.hessian.utilities import copy_wts_into_model
    except ImportError:
        pytest.skip("Core functions not available; skipping related tests.")
    
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)

    model_wrapper = SimpleModelWrapper(model)
    criterion = nn.MSELoss()
    loss_metric = Loss(criterion, X, y)
    
    # Create a random direction
    direction = [torch.randn_like(p) for p in model.parameters()]
    
    # Wrap direction to have the same shape as model parameters
    model_1 = SimpleModelWrapper(copy_wts_into_model(direction, model))

    # Generate a linear interpolation
    line = linear_interpolation(model_wrapper, model_1, loss_metric, distance=1, steps=5)

    # Check that the line is a list with correct dimensions
    assert isinstance(line, np.ndarray)
    assert len(line) == 5
    for point in line:
        assert isinstance(point, float)


def test_point():
    """Test the point function."""
    try:
        from oxford_loss_landscapes import point
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics import Loss
    except ImportError:
        pytest.skip("Core functions not available; skipping related tests.")
    
    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.randn(20, 3)
    y = torch.randn(20, 1)

    model_wrapper = SimpleModelWrapper(model)
    criterion = nn.MSELoss()
    loss_metric = Loss(criterion, X, y)

    # Get the loss
    y_pred = model(X)
    initial_loss = criterion(y_pred, y).item()

    # Compute the point value
    loss_value = point(model_wrapper, loss_metric)

    # Check that the loss value is a float
    assert isinstance(loss_value, float)
    assert np.isclose(loss_value, initial_loss)