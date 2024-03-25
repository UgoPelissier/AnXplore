import meshio
import numpy as np
from typing import Union, List

def tetra_volume(
        tetrahedron: np.ndarray
) -> np.ndarray :
    """
    Return the volume of an arrray of tetrahedrons.
    """
    if tetrahedron.ndim==2:
        s = tetrahedron.shape
        tetrahedron = tetrahedron.reshape((1, s[0], s[1]))
    return abs(np.linalg.det(np.swapaxes(np.stack((tetrahedron[:,0]-tetrahedron[:,3], tetrahedron[:,1]-tetrahedron[:,3], tetrahedron[:,2]-tetrahedron[:,3])), 0, 1)))/6

def volume_region(
        points: np.ndarray,
        cells: np.ndarray
) -> float:
    """
    Return the volume of a mesh region indicated by its cells (tetrahedra).
    """
    return np.sum(tetra_volume(points[cells]))

def clip_tetra(
        points: np.ndarray,
        cells: np.ndarray,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> tuple[np.ndarray]:
    """
    Return the cells cut by a plane.
    """
    return cells[np.nonzero(1*(np.sum(1*(np.dot(points-origin, normal)>0)[cells], axis=1)==4))[0]], cells[np.nonzero(1*(np.sum(1*(np.dot(points-origin, normal)>0)[cells], axis=1)==0))[0]]