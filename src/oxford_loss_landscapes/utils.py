import numpy as np
import scipy

def estimate_surf_vol(x_range: tuple, y_range: tuple, z_array: np.array):
    """
    wrapper function for the trapezoidal_area function
    to match the signature of other functions in this file.
    """
    return trapezoidal_area(x_range, y_range, z_array)


def move_landscape_to_cpu(gpu_loss_landscape):
    """
    Convert a loss landscape computed on GPU (with PyTorch tensors) to a CPU-compatible format (nested lists of floats).
    """
    cpu_loss_landscape = []

    for row in gpu_loss_landscape:
        tmp_row = []
        for itm in row:
            itm = (float(itm[7:-1]))
            tmp_row.append(itm)
        cpu_loss_landscape.append(tmp_row)
    return cpu_loss_landscape


def trapezoidal_area(x_range: tuple, y_range: tuple, z_array: np.array):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    num_x_steps, num_y_steps = z_array.shape
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Fix: Use linspace to create exactly the right number of points to match z_array
    X = np.linspace(x_min, x_max, num_x_steps)
    Y = np.linspace(y_min, y_max, num_y_steps)
    X, Y = np.meshgrid(X, Y)
    Z = z_array

    xyz = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.simplices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    vol = np.cross(a, b) @ tri[:,:,2]
    return vol.sum() / 6.0