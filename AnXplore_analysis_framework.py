import os.path as osp
import meshio
import numpy as np
from typing import Union, List

def min_max_tawss(
        tawss: np.ndarray
)-> tuple[float, float]:
    """
    Return the minimum and maximum TAWSS values of the mesh.
    """
    min_tawss = np.min(tawss)
    max_tawss = np.max(tawss)
    return min_tawss, max_tawss

def mean_std(
        field: np.ndarray
)-> tuple[float, float]:
    """
    Return the mean and standard deviation of the TAWSS values of the mesh.
    """
    mean_tawss = np.mean(field)
    std_tawss = np.std(field)
    return mean_tawss, std_tawss

def split_vessels_aneurysm(
        mesh: meshio.Mesh,
        origin: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray]
)-> tuple[np.ndarray, np.ndarray]:
    """
    Split the mesh into two volumes: the vessels and the aneurysm.
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
    return vessels, aneurysm

def extract_surface_points(
        tawss: np.ndarray,
        region: np.ndarray,
) -> np.ndarray:
    """
    Extract the surface points of a region.
    """
    surface = []
    for i in region:
        if (tawss[i] > 1e-10):
            surface.append(i)
    surface = np.array(surface)
    return surface

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

def WSS_regions(
        aneurysm_surface: np.ndarray,
        tawss: np.ndarray,
        mean_tawss_vessels: float,
        std_tawss_vessels: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the WSS regions of the aneurysm as point data.
    """
    WSS_low = []
    WSS_high = []
    for i in aneurysm_surface:
        if (tawss[i] > mean_tawss_vessels + std_tawss_vessels):
            WSS_low.append(i)
        elif (tawss[i] < mean_tawss_vessels - std_tawss_vessels):
            WSS_high.append(i)
    WSS_low = np.array(WSS_low)
    WSS_high = np.array(WSS_high)
    regions = np.zeros(len(tawss))
    regions[WSS_high] = 1
    regions[WSS_low] = -1
    return WSS_low, WSS_high, regions

data_dir = "data/"
filename = "AnXplore178_FSI_00045.vtu"

origin = [0.0, 5.5, 0.0]
plane = [0.0, 1.0, 0.0]

if __name__ == '__main__':
    # Load the mesh
    path = osp.join(data_dir, filename)
    mesh = meshio.read(path)
    print(f"\nReading {filename}...\n{mesh}\n")

    # Print the minimum and maximum TAWSS values
    min_tawss, max_tawss = min_max_tawss(mesh.point_data['TAWSS'])

    # Extract the aneurysm points and print the minimum and maximum TAWSS values
    vessels, aneurysm = split_vessels_aneurysm(mesh, origin, plane)
    vessels_surface = extract_surface_points(mesh.point_data['TAWSS'], vessels)
    aneurysm_surface = extract_surface_points(mesh.point_data['TAWSS'], aneurysm)
    min_tawss_aneurysm, max_tawss_aneurysm = min_max_tawss(mesh.point_data['TAWSS'][aneurysm_surface])

    # Mean and std of TAWSS in the vessels
    mean_tawss_vessels, std_tawss_vessels = mean_std(mesh.point_data['TAWSS'][vessels_surface])

    # Extract the surface cells
    surface_cells = extract_surface_cells(mesh)

    # WSS regions (point data)
    WSS_low, WSS_high, regions = WSS_regions(aneurysm_surface, mesh.point_data['TAWSS'], mean_tawss_vessels, std_tawss_vessels)

    # Print the results
    print(f"min_tawss = {min_tawss}\nmax_tawss = {max_tawss}\n")
    print(f"min_tawss_aneurysm = {min_tawss_aneurysm}\nmax_tawss_aneurysm = {max_tawss_aneurysm}\n")
    print(f"mean_tawss_vessels = {mean_tawss_vessels}\nstd_tawss_vessels = {std_tawss_vessels}\n")

    # Save the results
    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", surface_cells)],
        point_data = {"regions": regions}
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_surface.vtu"), mesh)
