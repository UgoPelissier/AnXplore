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
        points: np.ndarray,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split points: aneurysm volume and vessels volume.
    """
    vessels = np.nonzero(1-1*(np.dot(points-origin, normal)>0))[0]
    aneurysm = np.nonzero(1*(np.dot(points-origin, normal)>0))[0]
    aneurysm_inidcator = np.zeros(len(points))
    aneurysm_inidcator[aneurysm] = 1
    return vessels, aneurysm, aneurysm_inidcator

def extract_surface_points(
        indicator: np.ndarray,
        volume: np.ndarray,
) -> np.ndarray:
    """
    Extract the points of a volume belonging to the surface.
    """
    return volume[(np.nonzero(1*(indicator[volume]>1e-10)))[0]]

def WSS_regions(
        wss: np.ndarray,
        mean_wss_vessels: float,
        std_wss_vessels: float
) -> np.ndarray:
    """
    Compute the WSS regions.
    """
    low = 1*(wss<(mean_wss_vessels-std_wss_vessels))
    high = 1*(wss>(mean_wss_vessels+std_wss_vessels))
    return high-low