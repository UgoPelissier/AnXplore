import meshio
import numpy as np

def normal(triangles):
    return np.cross(triangles[:,1] - triangles[:,0], 
                    triangles[:,2] - triangles[:,0], axis=1)

def triangle_area(triangles: np.ndarray) -> np.ndarray:
    """
    Return the area of a an array of triangles.
    """
    return np.linalg.norm(normal(triangles), axis=1) / 2

def surface_area(
        points: np.ndarray,
        cells: np.ndarray
)-> float:
    """
    Return the surface area of a mesh region indicated by its cells (triangles).
    """
    return np.sum(triangle_area(points[cells]))

def extract_surface_cells(
        cells: np.ndarray,
        wss: np.ndarray
) -> np.ndarray:
    """
    Extract the surface cells of a mesh.
    """
    x, y = np.nonzero(1*wss[cells[np.nonzero(np.sum(wss[cells]>0, axis=1)==3)[0]]]>0)
    return cells[np.nonzero(np.sum(wss[cells]>0, axis=1)==3)[0]][x,y].reshape(-1,3)

def aneurysm_cells(
        surface_cells: np.ndarray,
        aneurysm_surface: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cells belonging to the aneurysm and vessels.
    """
    return surface_cells[np.nonzero(1*(np.sum(aneurysm_surface[surface_cells], axis=1)==3))[0],:], surface_cells[np.nonzero(1*(np.sum(aneurysm_surface[surface_cells], axis=1)!=3))[0],:]

def WSS_regions_cells(
        surface_cells: np.ndarray,
        regions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cells belonging to the WSS low and high regions.
    """
    return surface_cells[np.nonzero(1*(np.sum(regions[surface_cells], axis=1)==-3))[0]], surface_cells[np.nonzero(1*(np.sum(regions[surface_cells], axis=1)==3))[0]]

def positive_v_y_cells(
        mesh: meshio.Mesh,
) -> np.ndarray:
    """
    Return the cells with a positive y component of the velocity.
    """
    return mesh.cells[0].data[np.nonzero(1*(np.sum(1*(mesh.point_data['Vitesse'][:,1][mesh.cells[0].data]>0), axis=1)==3))[0],:]