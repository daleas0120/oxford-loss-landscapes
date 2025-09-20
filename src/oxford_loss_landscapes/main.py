"""
Functions for approximating loss/return landscapes in one and two dimensions.
"""
import copy
import typing
import torch.nn
import numpy as np
import os
import datetime
from tqdm import trange
try:
    import ray # for parallel processing
except ImportError:
    ray = None
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from .model_interface.model_wrapper import ModelWrapper, wrap_model
from .model_interface.model_parameters import rand_n_like, orthogonal_to
from .metrics.metric import Metric
from .hessian.utilities import copy_wts_into_model

USE_PARALLEL = False

def _evaluate_plane(start_point, dir_one, dir_two, steps, metric, model_wrapper, distance=1.0, export=False):
    """
    Helper function to evaluate a planar region in parameter space.
    This avoids code duplication between planar_interpolation and random_plane.
    Exports binary landscape output and config toml to results folder if export is True, else outputs numpy array.
    """
    data_matrix = []
    # evaluate loss in grid of (steps * steps) points, where each column signifies one step
    # along dir_one and each row signifies one step along dir_two. The implementation is again
    # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
    # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].

    # We need to work directly with the model parameters to avoid gradient issues
    # Get reference to the actual model parameters
    model_params = model_wrapper.get_module_parameters()
    
    for i in trange(steps, desc='Calculating Surface...'):
        data_column = []

        for _ in range(steps):
            # for every other column, reverse the order in which the column is generated
            # Directly modify the model parameters using non-in-place operations
            if i % 2 == 0:
                # Update model parameters: model_params = model_params + dir_two
                for idx in range(len(model_params)):
                    model_params.parameters[idx].data = model_params[idx].data + dir_two[idx].data
                data_column.append(metric(model_wrapper))
            else:
                # Update model parameters: model_params = model_params - dir_two  
                for idx in range(len(model_params)):
                    model_params.parameters[idx].data = model_params[idx].data - dir_two[idx].data
                data_column.insert(0, metric(model_wrapper))

        data_matrix.append(data_column)
        # Move to next row: model_params = model_params + dir_one
        for idx in range(len(model_params)):
            model_params.parameters[idx].data = model_params[idx].data + dir_one[idx].data

    if export:
        _export_plane_to_npy(data_matrix, distance)
    else:
        return np.array(data_matrix)


def _evaluate_plane_parallel(start_point, dir_one, dir_two, steps, metric, model_wrapper, distance=1.0, export=False, num_workers=None):
    """
    Ray-based parallel version of _evaluate_plane that distributes row calculations across workers.
    
    This function parallelizes the outer loop by computing each row (column in the matrix) 
    independently using Ray remote functions. Falls back to sequential evaluation if Ray 
    fails to initialize or execute.
    
    :param start_point: Initial point in parameter space
    :param dir_one: Direction vector for row steps (outer loop)
    :param dir_two: Direction vector for column steps (inner loop) 
    :param steps: Number of steps in each direction
    :param metric: Function to evaluate model at each point
    :param model_wrapper: Model wrapper object
    :param distance: Distance parameter for export
    :param export: Whether to export results to file
    :param num_workers: Number of Ray workers (defaults to CPU count)
    :return: numpy array of evaluation results or None if exported
    """
    try:
        if ray is None:
            raise ImportError("Ray not available")
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Determine number of workers
        if num_workers is None:
            try:
                available_cpus = int(ray.available_resources().get("CPU", multiprocessing.cpu_count()))
                num_workers = min(steps, max(1, available_cpus))
            except:
                num_workers = min(steps, multiprocessing.cpu_count())

        @ray.remote
        def evaluate_row(row_start_point, d2, steps_inner, metric_func, wrapper, row_idx):
            """
            Evaluate a single row of the plane in parallel.
            
            :param row_start_point: Starting point for this row
            :param d1: Direction vector one (not used in inner loop)
            :param d2: Direction vector two (for inner loop steps)
            :param steps_inner: Number of steps in the inner loop
            :param metric_func: Metric function to evaluate
            :param wrapper: Model wrapper (will be deep copied)
            :param row_idx: Row index for snake-like traversal pattern
            :return: List of metric values for this row
            """
            import copy
            import torch
            
            # Clone tensors to avoid deepcopy issues with gradients
            # Convert to CPU and detach from computation graph for serialization
            current_point_data = []
            for param in row_start_point.parameters:
                current_point_data.append(param.data.clone().detach().cpu())
            
            d2_data = []
            for param in d2.parameters:
                d2_data.append(param.data.clone().detach().cpu())
            
            # Create independent model wrapper copy
            local_wrapper = copy.deepcopy(wrapper)
            local_model_params = local_wrapper.get_module_parameters()
            
            # Set initial parameters correctly based on the snake pattern
            # Sequential version: move first, then evaluate
            for idx in range(len(local_model_params)):
                local_model_params.parameters[idx].data = current_point_data[idx].clone()
            
            data_column = []
            
            for step in range(steps_inner):
                # Implement the same snake-like traversal pattern as original
                if row_idx % 2 == 0:
                    # Even rows: traverse forward
                    # Update model parameters: current_point = current_point + d2
                    for idx in range(len(local_model_params)):
                        local_model_params.parameters[idx].data = local_model_params[idx].data + d2_data[idx]
                    data_column.append(metric_func(local_wrapper))
                else:
                    # Odd rows: traverse backward 
                    # Update model parameters: current_point = current_point - d2
                    for idx in range(len(local_model_params)):
                        local_model_params.parameters[idx].data = local_model_params[idx].data - d2_data[idx]
                    value = metric_func(local_wrapper)
                    data_column.insert(0, value)
            
            return data_column

        # Prepare tasks for parallel execution
        tasks = []
        
        # Clone direction vector to avoid deepcopy issues
        d2_cloned = start_point.__class__(
            [param.data.clone().detach().cpu() for param in dir_two.parameters]
        )
        
        # Calculate starting positions for each row to match sequential traversal
        for i in range(steps):
            # Calculate where this row should start based on sequential logic
            if i == 0:
                # Row 0: start at original start_point
                row_start_pos = copy.deepcopy(start_point)
            else:
                # For subsequent rows, calculate based on where sequential version would be
                row_start_pos = copy.deepcopy(start_point)
                # Move down i rows  
                row_start_pos = row_start_pos + dir_one * i
                
                # For odd rows, also need to account for the snake pattern
                # The sequential version reaches odd rows at their rightmost position
                if i % 2 == 1:
                    row_start_pos = row_start_pos + dir_two * steps
            
            row_start = start_point.__class__(
                [param.data.clone().detach().cpu() for param in row_start_pos.parameters]
            )
            
            # Submit remote task for each row
            task = evaluate_row.remote(
                row_start, 
                d2_cloned, 
                steps, 
                metric, 
                model_wrapper, 
                i
            )
            tasks.append(task)
        
        # Collect results from all workers
        print(f"Computing loss landscape using {num_workers} Ray workers...")
        data_matrix = ray.get(tasks)
        
        # Convert to numpy array
        result = np.array(data_matrix)
        
        if export:
            _export_plane_to_npy(result, distance)
            return None
        else:
            return result
            
    except Exception as e:
        print(f"Ray parallel execution failed: {e}")
        print("Falling back to sequential evaluation...")
        
        # Fallback to original sequential implementation
        return _evaluate_plane(start_point, dir_one, dir_two, steps, metric, model_wrapper, distance, export)


def point(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric) -> tuple:
    """
    Returns the computed value of the evaluation function applied to the model
    or agent at a specific point in parameter space.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric
    class, and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    The model supplied can be either a torch.nn.Module model, or a ModelWrapper from the
    loss_landscapes library for more complex cases.

    :param model: the model or model wrapper defining the point in parameter space
    :param metric: Metric object used to evaluate model
    :return: quantity specified by Metric at point in parameter space
    """
    return metric(wrap_model(model))


def linear_interpolation(model_start: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end: typing.Union[torch.nn.Module, ModelWrapper],
                         metric: Metric, steps=100, deepcopy_model=False, distance=1.0) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or
    agent along a linear subspace of the parameter space defined by two end points.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given two models, for both of which the model's parameters define a
    vertex in parameter space, the evaluation is computed at the given number of steps
    along the straight line connecting the two vertices. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line, thus obtaining a view of the "straight line" in
    parameter space from the initialization to some minima. There is no guarantee
    that the model followed this path during optimization. In fact, it is highly
    unlikely to have done so, unless the optimization problem is convex.

    Note that a simple linear interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    use random_line() with filter normalization instead.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the start point of the line in parameter space
    :param model_end: the model defining the end point of the line in parameter space
    :param metric: list of function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    # create wrappers from deep copies to avoid aliasing if desired
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    end_model_wrapper = wrap_model(copy.deepcopy(model_end) if deepcopy_model else model_end)

    start_point = model_start_wrapper.get_module_parameters()
    # end_point = distance*end_model_wrapper.get_module_parameters()
    direction = distance*end_model_wrapper.get_module_parameters()

    direction.mul_(steps / 2)
    start_point.sub_(direction)
    direction.truediv_(steps / 2)

    data_values = []
    for _ in trange(steps, desc='Calculating...'):
        # add a step along the line to the model parameters, then evaluate
        start_point.add_(direction)
        data_values.append(metric(model_start_wrapper))

    return np.array(data_values)


def random_line(model_start: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric, distance=0.1, steps=100,
                normalization='filter', deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along a
    linear subspace of the parameter space defined by a start point and a randomly sampled direction.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the evaluation is computed at 'steps' points along a random
    direction, from the start point up to the maximum distance from the start point.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled other in parameter space. That is, the more parameters
    a model has, the longer the distance in the random other's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction other according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction other.

    Note also that a simple line approximation can produce misleading views
    of the loss landscape due to the scale invariance of neural networks. The sharpness or
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    normalize the direction, preferably with the 'filter' option.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: model to be evaluated, whose current parameters represent the start point
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the randomly sampled direction
    """
    # create wrappers from deep copies to avoid aliasing if desired
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)

    # obtain start point in parameter space and random direction
    # random direction is randomly sampled, then normalized, and finally scaled by distance/steps
    start_point = model_start_wrapper.get_module_parameters()
    direction = rand_n_like(start_point)

    if normalization == 'model':
        direction.model_normalize_(start_point)
    elif normalization == 'layer':
        direction.layer_normalize_(start_point)
    elif normalization == 'filter':
        direction.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    # direction.mul_(((start_point.model_norm() * distance) / steps) / direction.model_norm())
    start_point.sub_(direction)
    direction.truediv_(steps / 2)

    data_values = []
    for _ in trange(steps, desc='Calculating...'):
        # add a step along the line to the model parameters, then evaluate
        start_point.add_(direction)
        data_values.append(metric(model_start_wrapper))

    return np.array(data_values)


def planar_interpolation(model_start: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end_one: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end_two: typing.Union[torch.nn.Module, ModelWrapper],
                         metric: Metric, distance=1.0, steps=20, deepcopy_model=False, eigen_models=False, use_ray=False, num_workers=None) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along
    a planar subspace of the parameter space defined by a start point and two end points.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given two models, for both of which the model's parameters define a
    vertex in parameter space, the loss is computed at the given number of steps
    along the straight line connecting the two vertices. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line, thus obtaining a view of the "straight line" in
    paramater space from the initialization to some minima. There is no guarantee
    that the model followed this path during optimization. In fact, it is highly
    unlikely to have done so, unless the optimization problem is convex.

    That is, given three neural network models, 'model_start', 'model_end_one', and
    'model_end_two', each of which defines a point in parameter space, the loss is
    computed at 'steps' * 'steps' points along the plane defined by the start vertex
    and the two vectors (end_one - start) and (end_two - start), up to the maximum
    distance in both directions. A common choice would be for two of the points to be
    the model after initialization, and the model after convergence. The third point
    could be another randomly initialized model, since in a high-dimensional space
    randomly sampled directions are most likely to be orthogonal.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the origin point of the plane in parameter space
    :param model_end_one: the model representing the end point of the first direction defining the plane
    :param model_end_two: the model representing the end point of the second direction defining the plane
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :param eigen_models: whether the end models represent eigenvectors rather than displacement vectors
    :param use_ray: whether to use Ray for parallel computation (faster for large grids)
    :param num_workers: number of Ray workers (defaults to CPU count)
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if deepcopy_model else model_end_one)
    model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if deepcopy_model else model_end_two)

    # compute direction vectors
    start_point = model_start_wrapper.get_module_parameters()

    if eigen_models:
        dir_one = distance*(model_end_one_wrapper.get_module_parameters()) / steps
        dir_two = distance*(model_end_two_wrapper.get_module_parameters()) / steps
    else:
        dir_one = distance*(model_end_one_wrapper.get_module_parameters() - start_point) / steps
        dir_two = distance*(model_end_two_wrapper.get_module_parameters() - start_point) / steps
    
    # scale to match steps and total distance
    # dir_one.mul_(((start_point.model_norm() * distance) / steps) / dir_one.model_norm())
    # dir_two.mul_(((start_point.model_norm() * distance) / steps) / dir_two.model_norm())
    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    # Choose between parallel and sequential evaluation
    if use_ray or USE_PARALLEL:
        return _evaluate_plane_parallel(start_point, dir_one, dir_two, steps, metric, model_start_wrapper, num_workers=num_workers)
    return _evaluate_plane(start_point, dir_one, dir_two, steps, metric, model_start_wrapper)


def random_plane(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric, distance=1, steps=20,
                 normalization='filter', deepcopy_model=False, export=False, use_ray=False, num_workers=None) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along a planar
    subspace of the parameter space defined by a start point and two randomly sampled directions.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the loss is computed at 'steps' * 'steps' points along the
    plane defined by the two random directions, from the start point up to the maximum
    distance in both directions.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled other in parameter space. That is, the more parameters
    a model has, the longer the distance in the random other's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction other according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction other.

    Note also that a simple planar approximation with randomly sampled directions can produce
    misleading approximations of the loss landscape due to the scale invariance of neural
    networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
    network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
    recommended to normalize the directions, preferably with the 'filter' option.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model: the model defining the origin point of the plane in parameter space
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :param export: whether to export the landscape data to a file
    :param use_ray: whether to use Ray for parallel computation (faster for large grids)
    :param num_workers: number of Ray workers (defaults to CPU count)
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model) if deepcopy_model else model)

    start_point = model_start_wrapper.get_module_parameters()
    dir_one = rand_n_like(start_point)
    dir_two = orthogonal_to(dir_one)

    if normalization == 'model':
        dir_one.model_normalize_(start_point)
        dir_two.model_normalize_(start_point)
    elif normalization == 'layer':
        dir_one.layer_normalize_(start_point)
        dir_two.layer_normalize_(start_point)
    elif normalization == 'filter':
        dir_one.filter_normalize_(start_point)
        dir_two.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    # scale to match steps and total distance
    dir_one.mul_(((start_point.model_norm() * distance) / steps) / dir_one.model_norm())
    dir_two.mul_(((start_point.model_norm() * distance) / steps) / dir_two.model_norm())
    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)
    
    # Choose between parallel and sequential evaluation
    if use_ray or USE_PARALLEL:
        return _evaluate_plane_parallel(start_point, dir_one, dir_two, steps, metric, model_start_wrapper, distance, export, num_workers)
    return _evaluate_plane(start_point, dir_one, dir_two, steps, metric, model_start_wrapper, distance, export)
def _export_plane_to_npy(data_matrix, distance):
    """
    Exports the data_matrix as a .npy binary file in a folder called 'results' in the current working directory.
    """
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_LOLxAI"
    
    # Save distance to toml manually
    toml_filename = f"{filename}.toml"
    with open(os.path.join(results_dir, toml_filename), "w", encoding="utf-8") as f:
        f.write(f'distance = {float(distance)}\n')
    
    # Save landscape data to npy
    data_filename = f"{filename}.npy"
    data_file_path = os.path.join(results_dir, data_filename)
    np.save(data_file_path, data_matrix)

    print(f"Saved plane data to {data_file_path}")


# def _evaluate_plane_parallel(start_point, dir_one, dir_two, steps, metric, model_wrapper,
#                             use_ray: bool = False, ray_init_kwargs: dict = None, num_workers: int = None):
#     """
#     Parameters
#     ----------
#     start_point : tensor-like
#         Initial point in parameter space. This object will be mutated during
#         sequential evaluation, so pass a copy if you need to preserve it.
#     dir_one : tensor-like
#         Direction vector for row steps (x-axis).
#     dir_two : tensor-like
#         Direction vector for column steps (y-axis).
#     steps : int
#         Number of steps in each direction (grid will be steps x steps).
#     metric : callable
#         Function of (model_wrapper) that evaluates the loss/score at the
#         current parameters of the model.
#     model_wrapper : object
#         Wrapper object providing model parameters and any other state required
#         by `metric`.
#     use_ray : bool, default=True
#         If True, attempt to use Ray actors for parallel execution.
#         If False or Ray is unavailable, fall back to sequential evaluation.
#     ray_init_kwargs : dict, optional
#         Passed to `ray.init()` if Ray must be initialized.
#     num_workers : int, optional
#         Number of Ray workers (actors) to create. Defaults to CPU count,
#         capped by `steps`.

#     Returns
#     -------
#     np.ndarray
#         Array of shape (steps, steps) containing metric values for the plane.
#         Row index corresponds to dir_one steps, column index to dir_two steps.

#     Notes
#     -----
#     - If Ray initialization or execution fails for any reason,
#       the function silently falls back to the sequential implementation.
#     - The traversal pattern (snake-like per row) is identical to the
#       original `_evaluate_plane` for consistency.
#     """
#     # sequential fallback implementation (identical behaviour to original)
#     def _sequential_eval(sp, d1, d2, steps_, metric_, wrapper_):
#         data_matrix = []
#         for i in range(int(steps_)):
#             data_column = []
#             for _ in range(int(steps_)):
#                 if i % 2 == 0:
#                     sp.add_(d2)
#                     data_column.append(metric_(wrapper_))
#                 else:
#                     sp.sub_(d2)
#                     data_column.insert(0, metric_(wrapper_))
#             data_matrix.append(data_column)
#             sp.add_(d1)
#         return np.array(data_matrix)

#     if not use_ray or ray is None:
#         return _sequential_eval(start_point, dir_one, dir_two, steps, metric, model_wrapper)

#     # initialize ray if needed
#     if not ray.is_initialized():
#         ray.init(**(ray_init_kwargs or {}))

#     # choose number of workers
#     try:
#         available_cpus = int(ray.available_resources().get("CPU", os.cpu_count() or 1))
#     except Exception:
#         available_cpus = os.cpu_count() or 1
#     if num_workers is None:
#         num_workers = min(steps, max(1, available_cpus))

#     @ray.remote
#     def eval_row(sp_row_start, dir_one_, dir_two_, steps_, metric_, wrapper_, row_idx):
#         """
#         Evaluates a single row of the plane, accounting for snake-like traversal.
#         """
#         current_point = copy.deepcopy(sp_row_start)
#         data_column = []
#         local_wrapper = copy.deepcopy(wrapper_)

#         # For odd rows, shift the starting point to the far right.
#         if row_idx % 2 != 0:
#             # Create a new tensor representing the total horizontal shift
#             # by multiplying the direction vector by the number of steps.
#             total_shift = dir_two_.mul_(steps_)
#             # Add this single, scaled vector to the starting point.
#             current_point.add_(total_shift)

#         # The rest of the logic remains the same
#         for _ in range(steps_):
#             if row_idx % 2 == 0:
#                 current_point.add_(dir_two_) # Traverse right
#             else:
#                 current_point.sub_(dir_two_) # Traverse left
            
#             local_wrapper.get_module_parameters() #TODO: need to copy parameters to local_wrapper
#             data_column.append(metric_(local_wrapper))
            
#         if row_idx % 2 != 0:
#             data_column.reverse()
            
#         return data_column

#     # The main loop now prepares the correct starting point for each row
#     ray_tasks = []
#     # Use a mutable copy for calculating row start points
#     current_sp = copy.deepcopy(start_point)

#     for i in range(int(steps)):
#         # Launch a Ray task for the current row's starting point
#         # Pass the original model_wrapper; the worker will deepcopy it.
#         ray_tasks.append(eval_row.remote(copy.deepcopy(current_sp), dir_one, dir_two, steps, metric, model_wrapper, i))
#         # Move the reference point for the next row's starting point
#         current_sp.add_(dir_one)
        
#     # collect results and assemble matrix (rows as returned)
#     try:
#         print("Warning: Parallel evaluation may produce incorrect results due to known issues with parameter copying and model state. It is recommended to use sequential evaluation for reliable results. For more information, see the documentation or report issues at https://github.com/oxford-loss-landscapes/issues.")
#         results = ray.get(ray_tasks)
#     except Exception:
#         # if remote execution fails for any reason, fallback to sequential
#         print("Warning: Ray remote execution failed, falling back to sequential evaluation.")
#         traceback.print_exc()
#         return _sequential_eval(start_point, dir_one, dir_two, steps, metric, model_wrapper)

#     return np.array(results)

