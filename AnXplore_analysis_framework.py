import os.path as osp
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
    Return the surface area of a mesh region indicated by its cells.
    """
    surface_area = 0
    for cell in cells:
        surface_area += triangle_area(mesh.points[cell])
    return surface_area

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
)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    aneurysm_inidcator = np.zeros(len(mesh.points))
    aneurysm_inidcator[aneurysm] = 1
    return vessels, aneurysm, aneurysm_inidcator

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

def aneurysm_cells(
        surface_cells: np.ndarray,
        aneurysm_surface: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cells belonging to the WSS low and high regions.
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
    vessels, aneurysm, aneurysm_inidcator = split_vessels_aneurysm(mesh, origin, plane)
    vessels_surface = extract_surface_points(mesh.point_data['TAWSS'], vessels)
    aneurysm_surface = extract_surface_points(mesh.point_data['TAWSS'], aneurysm)
    min_tawss_aneurysm, max_tawss_aneurysm = min_max_tawss(mesh.point_data['TAWSS'][aneurysm_surface])

    # Mean and std of TAWSS in the vessels
    mean_tawss_vessels, std_tawss_vessels = mean_std(mesh.point_data['TAWSS'][vessels_surface])

    # Extract the surface cells
    surface_cells = extract_surface_cells(mesh)

    # WSS regions (point data)
    WSS_low, WSS_high, regions = WSS_regions(aneurysm_surface, mesh.point_data['TAWSS'], mean_tawss_vessels, std_tawss_vessels)

    # Compute area of the vessels and aneurysm
    mesh_area = surface_area(mesh, surface_cells)
    aneurysm_surface_cells, vessels_surface_cells = aneurysm_cells(surface_cells, aneurysm_inidcator)
    aneurysm_area = surface_area(mesh, aneurysm_surface_cells)
    vessels_area = surface_area(mesh, vessels_surface_cells)
    WSS_low_cells, WSS_high_cells = WSS_regions_cells(surface_cells, regions)
    WSS_low_area = surface_area(mesh, WSS_low_cells)
    WSS_high_area = surface_area(mesh, WSS_high_cells)

    # Print the results
    print(f"min_tawss = {min_tawss}\nmax_tawss = {max_tawss}\n")
    print(f"min_tawss_aneurysm = {min_tawss_aneurysm}\nmax_tawss_aneurysm = {max_tawss_aneurysm}\n")
    print(f"mean_tawss_vessels = {mean_tawss_vessels}\nstd_tawss_vessels = {std_tawss_vessels}\n")
    print(f"mesh_area = {mesh_area}\naneurysm_area = {aneurysm_area}\nvessels_area = {vessels_area}\n")
    print(f"aneurysm_WSS_low_area = {WSS_low_area}\naneurysm_WSS_high_area = {WSS_high_area}\nLSA = {100*WSS_low_area/aneurysm_area:.1f}%\n")

    # Save the results
    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", surface_cells)],
        point_data = {"aneurysm": aneurysm_inidcator, "regions": regions}
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_surface.vtu"), mesh)
