import numpy as np
import scipy

def estimate_surf_vol(x_range: tuple, y_range: tuple, z_array: np.array):
    assert len(z_array.shape) == 2, 'z_array is not 2D'
    num_x_steps, num_y_steps = z_array.shape

    x_min, x_max = x_range
    y_min, y_max = y_range

    x_interval = np.linspace(x_min, x_max, num_x_steps, endpoint=True)
    y_interval = np.linspace(y_min, y_max, num_y_steps, endpoint=True)

    z_vol = np.trapz(x_interval, np.trapz(y_interval, z_array))

    return z_vol

def move_landscape_to_cpu(gpu_loss_landscape):
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

    X = np.arange(start=x_min, stop=x_max, step=1/num_x_steps)
    Y = np.arange(start=y_min, stop=y_max, step=1/num_y_steps)
    X, Y = np.meshgrid(X, Y)
    Z = z_array

    xyz = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.simplices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    vol = np.cross(a, b) @ tri[:,:,2]
    return vol.sum() / 6.0