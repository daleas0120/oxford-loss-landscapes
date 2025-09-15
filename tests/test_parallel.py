import time
import copy
import types
import numpy as np
import numpy.testing as npt

from oxford_loss_landscapes import main


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


def _make_fake_plane_worker():
    """
    Create a FakePlaneWorker factory compatible with main._evaluate_plane_parallel usage:
    - has a .remote(...) classmethod returning an actor-like object with eval_plane_row.remote(...)
    """
    def actor_factory(model_wrapper, metric):
        # keep a deep copy like the real actor does
        local_wrapper = copy.deepcopy(model_wrapper)
        def eval_plane_row_remote(dir_one, dir_two, row_idx, steps):
            start_p = local_wrapper.get_module_parameters()
            # reproduce initialization used in parallel_utils.PlaneWorker
            start_p.sub_(dir_one)
            start_p.sub_(dir_two)
            for _ in range(int(row_idx)):
                start_p.add_(dir_one)

            data_column = []
            for j in range(int(steps)):
                if j % 2 == 0:
                    start_p.add_(dir_two)
                    data_column.append(metric(local_wrapper))
                else:
                    start_p.sub_(dir_two)
                    data_column.insert(0, metric(local_wrapper))
            return data_column

        actor = types.SimpleNamespace()
        actor.eval_plane_row = types.SimpleNamespace(remote=eval_plane_row_remote)
        return actor

    fake_cls = types.SimpleNamespace(remote=staticmethod(actor_factory))
    return fake_cls

#@pytest.mark.skipif("ray" not in globals(), reason="Ray not available")
def test_serial_and_parallel_plane_equal_and_timing():
    """
    Verify that the serial _evaluate_plane and the parallel implementation in main
    return the same results on a small grid. Also print timing for comparison.
    """
    steps = 3000
    dir_one = Delta(0.7)
    dir_two = Delta(0.3)

    # SERIAL
    wrapper_serial = DummyModelWrapper(0.0)
    start_serial = wrapper_serial.get_module_parameters()

    t0 = time.time()
    serial_res = main._evaluate_plane(start_serial, dir_one, dir_two, steps, metric_identity, wrapper_serial)
    t_serial = time.time() - t0
    print(f"Serial time: {t_serial:.6f}s")

    wrapper_parallel = DummyModelWrapper(0.0)
    start_parallel = wrapper_parallel.get_module_parameters()

    t0 = time.time()
    parallel_res = main._evaluate_plane_parallel(start_parallel, dir_one, dir_two, steps,
                                                    metric_identity, wrapper_parallel,
                                                    use_ray=True, ray_init_kwargs=None, num_workers=3)
    t_parallel = time.time() - t0
    print(f"Parallel (simulated) time: {t_parallel:.6f}s")


    # Both should be numpy arrays with same shape and values
    serial_arr = np.asarray(serial_res)
    parallel_arr = np.asarray(parallel_res)
    assert serial_arr.shape == parallel_arr.shape
    npt.assert_allclose(serial_arr, parallel_arr, rtol=1e-7, atol=1e-9)