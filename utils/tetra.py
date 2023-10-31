import meshio
import numpy as np
from typing import Union, List

def tetra_volume(
        tetrahedron: np.ndarray
) -> float:
    """
    Return the volume of a tetrahedron.
    """
    return abs(np.linalg.det(np.array([tetrahedron[0]-tetrahedron[3], tetrahedron[1]-tetrahedron[3], tetrahedron[2]-tetrahedron[3]])))/6

def volume_region(
        points: np.ndarray,
        cells: np.ndarray
) -> float:
    """
    Return the volume of a mesh region indicated by its cells (tetrahedra).
    """
    volume = 0
    for cell in cells:
        volume += tetra_volume(points[cell])
    return volume

def slice_tetra(
        points: np.ndarray,
        cells: np.ndarray,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> tuple[np.ndarray]:
    """
    Return the cells cut by a plane.
    """
    return cells[np.nonzero(1*(np.sum(1*(np.dot(points-origin, normal)>0)[cells], axis=1)==4))[0]], cells[np.nonzero(1*(np.sum(1*(np.dot(points-origin, normal)>0)[cells], axis=1)==0))[0]]