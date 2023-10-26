import meshio
import numpy as np
from typing import Union, List
from utils.cells import triangle_area

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