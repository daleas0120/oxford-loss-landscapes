import time
import copy
import types
import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn

from oxford_loss_landscapes import main
from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper, wrap_model
from oxford_loss_landscapes.metrics import Loss
from oxford_loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to


class DummyModelWrapper:
    """Minimal model wrapper supporting deepcopy and get_module_parameters."""
    def __init__(self, value: float):
        self.value = float(value)

    def get_module_parameters(self):
        return _ParamsView(self)

    def __deepcopy__(self, memo):
        return DummyModelWrapper(copy.deepcopy(self.value, memo))


class _ParamsView:
    """Provides add_/sub_ to mutate the underlying wrapper.value like tensors do."""
    def __init__(self, wrapper: DummyModelWrapper):
        self._wrapper = wrapper

    def _as_number(self, delta):
        # Prefer Delta.amount if present
        if hasattr(delta, "amount"):
            return float(delta.amount)
        # Try direct float conversion (works for Python numbers and numpy scalars)
        try:
            return float(delta)
        except Exception:
            pass
        # Try .item() (works for many tensor-like/np types)
        try:
            return float(delta.item())
        except Exception:
            pass
        raise TypeError(f"Cannot convert delta to float: {type(delta)}")

    def add_(self, delta):
        self._wrapper.value += self._as_number(delta)

    def sub_(self, delta):
        self._wrapper.value -= self._as_number(delta)


class Delta:
    """Simple delta object to emulate tensor-like steps."""
    def __init__(self, amount: float):
        self.amount = float(amount)


def metric_identity(wrapper: DummyModelWrapper):
    """Metric that returns the current scalar parameter value."""
    return float(wrapper.value)


def create_simple_model():
    """Create a simple neural network for demonstration."""
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model


def generate_data(n_samples=100):
    """Generate simple synthetic data for demonstration."""
    # Generate 2D input data
    X = torch.randn(n_samples, 2)
    # Simple target: sum of squares
    y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def test_evaluate_plane_parallel():
    """Test the parallel plane evaluation function with a dummy model wrapper."""
    try:
        from oxford_loss_landscapes.main import _evaluate_plane_parallel
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper, wrap_model
        from oxford_loss_landscapes.metrics import Loss
        from oxford_loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to
    except ImportError:
        print("Could not import _evaluate_plane_parallel, skipping test.")
        return

    # Create model and data
    model = create_simple_model()
    X, y = generate_data()
    criterion = nn.MSELoss()

    # Wrap the model
    model_wrapper = SimpleModelWrapper(copy.deepcopy(model))
    start_point = SimpleModelWrapper(model).get_module_parameters()
    dir_one = rand_u_like(start_point)
    dir_two = orthogonal_to(dir_one)

    # Create a metric to evaluate loss
    loss_metric = Loss(criterion, X, y)

    plane = _evaluate_plane_parallel(
        start_point=start_point,
        dir_one=dir_one,
        dir_two=dir_two,
        steps=10,
        metric=loss_metric,
        model_wrapper=model_wrapper,
        distance=1.0, export=False,
        num_workers=4
    )

    assert isinstance(plane, np.ndarray)
    assert plane.shape == (10, 10)
    assert np.all(np.isfinite(plane))
    assert np.all(plane >= 0.0)  # Loss should be non-negative


