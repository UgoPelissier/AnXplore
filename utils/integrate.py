import numpy as np
from utils.triangles import triangle_area
from utils.tetra import tetra_volume

def integrate_field_surface(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a surface. """
    return np.sum(triangle_area(points[cells])*(np.sum(data[cells],axis=1)/3))

def integrate_field_volume(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a volume. """
    return np.sum(tetra_volume(points[cells])*(np.sum(data[cells],axis=1)/4))