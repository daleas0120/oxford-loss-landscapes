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
        nn.Linear(100, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    return model


def create_simple_model():
    """Create a simple neural network for demonstration."""
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model


def generate_data(n_samples=100):
    """Generate simple synthetic data for demonstration."""
    # Generate 2D input data
    X = torch.randn(n_samples, 2)
    # Simple target: sum of squares
    y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1) + 0.1 * torch.randn(n_samples, 1)
    return X, y

def test_serial_and_parallel_plane_equal_and_timing2():
    """
    Verify that the serial _evaluate_plane and the parallel implementation in main
    return the same results on a small grid. Also print timing for comparison.
    """
    # Create model and data
    print("1. Creating model and data...")
    model = create_simple_model()
    X, y = generate_data()
    criterion = nn.MSELoss()
    
    # Wrap the model
    print("2. Wrapping model with loss landscape interface...")
    model_wrapper = SimpleModelWrapper(model)
    
    # Test forward pass
    print("3. Testing forward pass...")
    with torch.no_grad():
        outputs = model_wrapper.forward(X)
        loss = criterion(outputs, y)
        print(f"   Initial loss: {loss.item():.4f}")
    
    # Create a metric to evaluate loss
    print("4. Creating loss metric...")
    loss_metric = Loss(criterion, X, y)

    steps = 10

    model_wrapper = wrap_model(copy.deepcopy(model))
    start_point = model_wrapper.get_module_parameters()
    dir_one = rand_u_like(start_point)
    dir_two = orthogonal_to(dir_one)

    dir_one.model_normalize_(start_point)
    dir_two.model_normalize_(start_point)


    # scale to match steps and total distance
    dir_one.mul_(((start_point.model_norm()) / steps) / dir_one.model_norm())
    dir_two.mul_(((start_point.model_norm()) / steps) / dir_two.model_norm())
    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)


    t0 = time.time()
    serial_res = main._evaluate_plane_parallel(start_point, dir_one, dir_two, steps, loss_metric, model_wrapper, use_ray=False)
    t_serial = time.time() - t0
    print(f"Serial time: {t_serial:.6f}s")


    t0 = time.time()
    parallel_res = main._evaluate_plane_parallel(start_point, dir_one, dir_two, steps, loss_metric, model_wrapper, use_ray=True)
    t_parallel = time.time() - t0
    print(f"Parallel time: {t_parallel:.6f}s")


    print(serial_res)
    print(parallel_res)

    # Both should be numpy arrays with same shape and values
    serial_arr = np.asarray(serial_res)
    parallel_arr = np.asarray(parallel_res)
    assert serial_arr.shape == parallel_arr.shape
    npt.assert_allclose(serial_arr, parallel_arr, rtol=1e-7, atol=1e-9)


# def test_serial_and_parallel_plane_equal_and_timing():
#     """
#     Verify that the serial _evaluate_plane and the parallel implementation in main
#     return the same results on a small grid. Also print timing for comparison.
#     """
#     steps = 3000
#     dir_one = Delta(0.7)
#     dir_two = Delta(0.3)

#     # SERIAL
#     wrapper_serial = DummyModelWrapper(0.0)
#     start_serial = wrapper_serial.get_module_parameters()

#     t0 = time.time()
#     serial_res = main._evaluate_plane(start_serial, dir_one, dir_two, steps, metric_identity, wrapper_serial)
#     t_serial = time.time() - t0
#     print(f"Serial time: {t_serial:.6f}s")

#     wrapper_parallel = DummyModelWrapper(0.0)
#     start_parallel = wrapper_parallel.get_module_parameters()

#     t0 = time.time()
#     parallel_res = main._evaluate_plane_parallel(start_parallel, dir_one, dir_two, steps,
#                                                     metric_identity, wrapper_parallel,
#                                                     use_ray=True, ray_init_kwargs=None, num_workers=3)
#     t_parallel = time.time() - t0
#     print(f"Parallel (simulated) time: {t_parallel:.6f}s")


#     # Both should be numpy arrays with same shape and values
#     serial_arr = np.asarray(serial_res)
#     parallel_arr = np.asarray(parallel_res)
#     assert serial_arr.shape == parallel_arr.shape
#     npt.assert_allclose(serial_arr, parallel_arr, rtol=1e-7, atol=1e-9)