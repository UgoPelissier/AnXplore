import vtk
import numpy as np
from typing import Union, List

from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import tri
from alive_progress import alive_bar
import meshio
import os.path as osp

class VTU_Wrapper(object):
    """
        Class used to read vtu files and performing cuts.
        The plane surfaces created by the cuts can be also

    """

    def __init__(self, filename: str) -> None:

        if not (filename.endswith(".vtu") or filename.endswith(".vtk")):
            filename += ".vtu"

        vtu_data = vtk.vtkXMLUnstructuredGridReader()
        vtu_data.SetFileName(filename)
        vtu_data.Update()

        self.filename = filename
        self.vtu_data = vtu_data

    def get_slice_data(
        self,
        center: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray],
        field: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Get slice of data from using a normal and an center for the slice.

        Input:
            center : List, tuple or nparray [x0, y0, z0]
            normal : List, tuple or nparray [nx, ny, nz]

        Output:
            coord : Numpy array of the coordinates.
                        x=coord[:,0]
                        y=coord[:,1]
                        z=coord[:,2]

            data : Numpy array of the coordinates.
        """

        plane = vtk.vtkPlane()
        plane.SetOrigin(*center)
        plane.SetNormal(*normal)

        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputConnection(self.vtu_data.GetOutputPort())
        cutter.Update()

        cut = cutter.GetOutput()

        # Coordinates of nodes in the mesh
        nodes_vtk_array = cut.GetPoints().GetData()
        coord = vtk_to_numpy(nodes_vtk_array)

        data = cut.GetPointData().GetArray(field)

        try:
            assert data
        except AssertionError:
            print("Error: Cut is empty!")
            exit(1)

        data = vtk_to_numpy(data)

        massProp = vtk.vtkMassProperties()
        massProp.SetInputData(cut)
        cut_area = massProp.GetSurfaceArea()

        self.cut = cut

        coord, idx = np.unique(coord, return_index=True, axis=0)

        return coord, data[idx], cut_area

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------


def integrate_vtk_surface(
        data_dir: str,
        filename: str,
        coord: np.ndarray,
        data: np.ndarray,
        field: str
) -> float:
    """ Integrate scalar p1 field over a surface. """
    dt = tri.Triangulation(coord[:,0], coord[:,2])

    surface = meshio.Mesh(
        points=coord,
        cells=[("triangle", dt.triangles)],
        point_data={field: data}
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_orifice.vtu"), surface)

    