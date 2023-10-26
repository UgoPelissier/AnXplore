import meshio
import numpy as np
from typing import Union, List

### Points ###

def four_points_to_triangles(
        points: np.ndarray
) -> np.ndarray:
    """
    Return the 4 triangles of a tetrhedron.
    """
    triangles = []
    triangles.append([points[0], points[1], points[2]])
    triangles.append([points[0], points[1], points[3]])
    triangles.append([points[0], points[2], points[3]])
    triangles.append([points[1], points[2], points[3]])
    return np.array(triangles)

def min_max(
        field: np.ndarray
)-> tuple[float, float]:
    """
    Return the minimum and maximum TAWSS values of the mesh.
    """
    min_field = np.min(field)
    max_field = np.max(field)
    return min_field, max_field

def mean_std(
        field: np.ndarray
)-> tuple[float, float]:
    """
    Return the mean and standard deviation of the TAWSS values of the mesh.
    """
    mean_field = np.mean(field)
    std_field = np.std(field)
    return mean_field, std_field

def project_point_on_plane(
        point: Union[List[float], np.ndarray],
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Project a point on a plane.
    """
    point = np.array(point)
    origin = np.array(origin)
    normal = np.array(normal)
    return point - np.dot(point-origin, normal) * normal

def split_vessels_aneurysm(
        mesh: meshio.Mesh,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split points: aneurysm volume and vessels volume.
    """
    vessels = []
    aneurysm = []
    for i, point in enumerate(mesh.points):
        if np.dot(point-origin, normal) > 0:
            aneurysm.append(i)
        else:
            vessels.append(i)
    vessels = np.array(vessels)
    aneurysm = np.array(aneurysm)
    aneurysm_inidcator = np.zeros(len(mesh.points))
    aneurysm_inidcator[aneurysm] = 1
    return vessels, aneurysm, aneurysm_inidcator

def extract_surface_points(
        indicator: np.ndarray,
        volume: np.ndarray,
) -> np.ndarray:
    """
    Extract the points of a volume belonging to the surface.
    """
    surface = []
    for i in volume:
        if (indicator[i] > 1e-10):
            surface.append(i)
    surface = np.array(surface)
    return surface

def WSS_regions(
        aneurysm_surface: np.ndarray,
        wss: np.ndarray,
        mean_wss_vessels: float,
        std_wss_vessels: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the WSS regions of the aneurysm as point data.
    """
    WSS_low = []
    WSS_high = []
    for i in aneurysm_surface:
        if (wss[i] > mean_wss_vessels + std_wss_vessels):
            WSS_high.append(i)
        elif (wss[i] < mean_wss_vessels - std_wss_vessels):
            WSS_low.append(i)
    WSS_low = np.array(WSS_low)
    WSS_high = np.array(WSS_high)
    regions = np.zeros(len(wss))
    regions[WSS_high] = 1
    regions[WSS_low] = -1
    return WSS_low, WSS_high, regions

def slice_tetra(
        mesh: meshio.Mesh,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Return the cells cut by a plane.
    """
    slice_tetra = []
    for cell in mesh.cells[0].data:
        dot_product = []
        for point in cell:
            dot_product.append(np.dot(mesh.points[point]-origin, normal))
        signed_dot_product = np.sign(dot_product)
        if (abs(np.sum(signed_dot_product)) != 4):
            slice_tetra.append(cell)
    slice_tetra = np.array(slice_tetra)
    return slice_tetra