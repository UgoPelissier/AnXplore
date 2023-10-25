import meshio
import numpy as np
from typing import Union, List

def triangle_area(
        triangle: np.ndarray
)-> float:
    """
    Return the area of a triangle.
    """
    a = np.linalg.norm(triangle[0]-triangle[1])
    b = np.linalg.norm(triangle[0]-triangle[2])
    c = np.linalg.norm(triangle[1]-triangle[2])
    s = (a + b + c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def surface_area(
        mesh: meshio.Mesh,
        cells: np.ndarray
)-> float:
    """
    Return the surface area of a mesh region indicated by its cells (triangles).
    """
    surface_area = 0
    for cell in cells:
        surface_area += triangle_area(mesh.points[cell])
    return surface_area

def extract_surface_cells(
        mesh: meshio.Mesh
) -> np.ndarray:
    """
    Extract the surface cells of a mesh.
    """
    surface_cells = []
    for cell in mesh.cells[0].data:
        triangle = []
        for point in cell:
            if (mesh.point_data['TAWSS'][point] > 1e-10):
                triangle.append(point)
        if (len(triangle) == 3):
            surface_cells.append(triangle)
    surface_cells = np.array(surface_cells)
    return surface_cells

def aneurysm_cells(
        surface_cells: np.ndarray,
        aneurysm_surface: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cells belonging to the aneurysm and vessels.
    """
    aneurysm_cells = []
    vessels_cells = []
    for cell in surface_cells:
        if (aneurysm_surface[cell[0]] == 1 and aneurysm_surface[cell[1]] == 1 and aneurysm_surface[cell[2]] == 1):
            aneurysm_cells.append(cell)
        else:
            vessels_cells.append(cell)
    aneurysm_cells = np.array(aneurysm_cells)
    vessels_cells = np.array(vessels_cells)
    return aneurysm_cells, vessels_cells

def WSS_regions_cells(
        surface_cells: np.ndarray,
        regions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cells belonging to the WSS low and high regions.
    """
    WSS_low_cells = []
    WSS_high_cells = []
    for cell in surface_cells:
        if (regions[cell[0]] == 1 and regions[cell[1]] == 1 and regions[cell[2]] == 1):
            WSS_high_cells.append(cell)
        elif (regions[cell[0]] == -1 and regions[cell[1]] == -1 and regions[cell[2]] == -1):
            WSS_low_cells.append(cell)
    WSS_low_cells = np.array(WSS_low_cells)
    WSS_high_cells = np.array(WSS_high_cells)
    return WSS_low_cells, WSS_high_cells