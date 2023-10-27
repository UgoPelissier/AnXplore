import numpy as np
from utils.triangles import triangle_area
from utils.tetra import tetra_volume

def integrate_field_surface(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a surface. """
    integral = 0
    for triangle in cells:
        integral += (data[triangle[0]] + data[triangle[1]] + data[triangle[2]])/3 * triangle_area(points[triangle])
    return integral

def integrate_field_volume(
        cells: np.ndarray,
        points: np.ndarray,
        data: np.ndarray,
) -> float:
    """ Integrate scalar p1 field over a volume. """
    integral = 0
    for tetra in cells:
        integral += (data[tetra[0]] + data[tetra[1]] + data[tetra[2]] + data[tetra[3]])/4 * tetra_volume(points[tetra])
    return integral