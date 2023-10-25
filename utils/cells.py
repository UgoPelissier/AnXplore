import meshio
import numpy as np
from typing import Union, List

from utils.points import four_points_to_triangles, slice_tetra

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
        points: np.ndarray,
        cells: np.ndarray
)-> float:
    """
    Return the surface area of a mesh region indicated by its cells (triangles).
    """
    surface_area = 0
    for cell in cells:
        surface_area += triangle_area(points[cell])
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

def slice_triangles(
        mesh: meshio.Mesh,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the triangles cut by a plane.
    """
    tetra = slice_tetra(mesh, origin, normal)

    slice_triangles = []
    slice_indicator = np.zeros(len(mesh.points))
    for tet in tetra:
        triangles = four_points_to_triangles(tet)
        for triangle in triangles:
            dot_product = []
            for point in triangle:
                dot_product.append(np.dot(mesh.points[point]-origin, normal))
            signed_dot_product = np.sign(dot_product)
            if (abs(np.sum(signed_dot_product)) != 3):
                slice_triangles.append(triangle)
                for point in triangle:
                    slice_indicator[point] = 1
    slice_triangles = np.array(slice_triangles)
    sliced_points = np.where(slice_indicator == 1)[0]
    return slice_triangles, slice_indicator, sliced_points

def positive_v_y_cells(
        mesh: meshio.Mesh,
        indicator: np.ndarray
) -> np.ndarray:
    """
    Return the cells with a positive y component of the velocity.
    """
    positive_v_y_cells = []
    for cell in mesh.cells[0].data:
        if (mesh.point_data['Vitesse'][cell[0]][1] > 0 and mesh.point_data['Vitesse'][cell[1]][1] > 0 and mesh.point_data['Vitesse'][cell[2]][1] > 0):
            positive_v_y_cells.append(cell)
    positive_v_y_cells = np.array(positive_v_y_cells)
    return positive_v_y_cells