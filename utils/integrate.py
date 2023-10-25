import meshio
import numpy as np
from typing import Union, List
from utils.cells import triangle_area

def integrate(
        mesh: meshio.Mesh,
        field: np.ndarray,
        surface_cells: np.ndarray
) -> float:
    """
    Integrate a field over a surface.
    """
    integral = 0
    for cell in surface_cells:
        integral += triangle_area(mesh.points[cell]) * (field[cell[0]] + field[cell[1]] + field[cell[2]]) / 3
    return integral