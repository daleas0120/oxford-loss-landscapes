import copy
import os
import ray



def initialize_ray(ray_init_kwargs: dict = None):
    """
    Initialize Ray if available and not already initialized.

    Parameters
    ray_init_kwargs : dict, optional
        Dictionary of keyword arguments passed directly to `ray.init()`.
        Examples include {'address': 'auto'} to connect to an existing cluster
        or {'num_cpus': 4} to restrict local resources.

    Returns
    -------
    bool
        True if Ray is available and either successfully initialized
        or already initialized; False if Ray is not available.
    """
    if not ray.is_initialized():
        ray.init(**(ray_init_kwargs or {}))

    return True


def choose_num_workers(num_workers: int = None, steps: int = None) -> int:
    """
    Choose a sensible default number of workers.
    - If num_workers provided and >=1, return that.
    - Else use available CPU count reported by Ray (if initialized) or os.cpu_count().
    - If steps is provided, cap workers at steps.
    """
    if num_workers is not None and num_workers >= 1:
        return int(num_workers)

    try:
        if ray is not None and ray.is_initialized():
            available = int(ray.available_resources().get("CPU", os.cpu_count() or 1))
        else:
            available = os.cpu_count() or 1
    except Exception:
        available = os.cpu_count() or 1

    workers = max(1, available)
    if steps is not None:
        workers = min(workers, int(steps))
    return workers


@ray.remote
class PlaneWorker:
    """
    Ray actor that keeps a deep copy of the provided model_wrapper and metric.
    It provides methods to evaluate rows of a 2D grid in parameter space.
    """

    def __init__(self, model_wrapper, metric):
        # keep local copies so parameter mutation doesn't affect other actors
        self.model_wrapper = copy.deepcopy(model_wrapper)
        self.metric = metric

    def eval_plane_row(self, dir_one, dir_two, row_idx, steps):
        """
        Compute a single row (indexed by row_idx) of the plane.
        Returns a Python list with 'steps' metric values (so ray.get returns serializable data).
        The method re-creates the start offset for the local parameters and then walks along dir_two.
        """
        # get a fresh reference to the parameters
        start_p = self.model_wrapper.get_module_parameters()
        # reproduce the same "centered" start used by main.planar_interpolation/random_plane:
        start_p.sub_(dir_one)
        start_p.sub_(dir_two)
        # move to requested row
        for _ in range(int(row_idx)):
            start_p.add_(dir_one)

        data_column = []
        for j in range(int(steps)):
            if j % 2 == 0:
                start_p.add_(dir_two)
                data_column.append(self.metric(self.model_wrapper))
            else:
                start_p.sub_(dir_two)
                data_column.insert(0, self.metric(self.model_wrapper))
        return data_column
