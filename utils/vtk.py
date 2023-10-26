import os.path as osp
from typing import Union, List, Optional
import numpy as np
import meshio
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import tri
from alive_progress import alive_bar

from utils.cells import triangle_area


class VTU_Wrapper(object):
    """
    Wrapper for the vtkXMLUnstructuredGridReader class.
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
        field: str,
        data_dir: str,
        filename: str
    ) -> tuple[meshio.Mesh, float]:
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
        data = data[idx]

        dt = tri.Triangulation(coord[:,0], coord[:,2])
        surface = meshio.Mesh(
            points=coord,
            cells=[("triangle", dt.triangles)],
            point_data={field: data}
        )
        meshio.write(osp.join(data_dir, filename), surface)

        return surface, cut_area