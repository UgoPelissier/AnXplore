import numpy as np
from utils.triangles import v_triangle_area
from utils.tetra import v_tetra_volume

def integrate_field_surface(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a surface. """
    return np.sum(v_triangle_area(points[cells])*(np.sum(data[cells],axis=1)/3))

def integrate_field_volume(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a volume. """
    return np.sum(v_tetra_volume(points[cells])*(np.sum(data[cells],axis=1)/4))