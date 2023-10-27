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
        mesh: meshio.Mesh,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> tuple[np.ndarray]:
    """
    Return the cells cut by a plane.
    """
    slice_up = []
    slice_down = []
    slice_tetra = []
    for cell in mesh.cells[0].data:
        dot_product = []
        for point in cell:
            dot_product.append(np.dot(mesh.points[point]-origin, normal))
        signed_dot_product = np.sign(dot_product)
        s = np.sum(signed_dot_product)
        if (s == 4):
            slice_up.append(cell)
        elif (s == -4):
            slice_down.append(cell)
        else:
            slice_tetra.append(cell)
    slice_up = np.array(slice_up)
    slice_down = np.array(slice_down)
    slice_tetra = np.array(slice_tetra)
    return slice_up, slice_down, slice_tetra