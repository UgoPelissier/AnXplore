
import os
import os.path as osp
from typing import Union, List
import numpy as np
import meshio
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import tri
from alive_progress import alive_bar

def decompress_h5(
        data_dir: str,
        filename: str
) -> None:
    # read the xdmf file
    os.makedirs(osp.join(data_dir, "vtu"), exist_ok=True)
    with meshio.xdmf.TimeSeriesReader(osp.join(data_dir, filename)) as reader:
        p, c = reader.read_points_cells()
        with alive_bar(reader.num_steps, title="Decompressing h5...") as bar:
            for t in range(reader.num_steps):
                _, P1, P0 = reader.read_data(t)
                mesh = meshio.Mesh(p,c,point_data=P1, cell_data=P0)
                mesh.write(osp.join(data_dir, "vtu", filename.split('.')[0]+"_{:05d}.vtu".format(t)))
                bar()


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
    
    def compute_vector_derivatives(
            self,
            field: str
    ) -> np.ndarray:
        """
        Compute the derivatives of a vector field.
        """
        cell_derivatives = vtk.vtkCellDerivatives()
        cell_derivatives.SetInputData(self.vtu_data.GetOutput())
        cell_derivatives.SetTensorModeToComputeGradient()
        cell_derivatives.SetVectorModeToPassVectors()
        self.vtu_data.GetOutput().GetPointData().SetActiveVectors(field)
        cell_derivatives.Update()
        vector_gradient = vtk_to_numpy(cell_derivatives.GetOutput().GetCellData().GetArray("VectorGradient")).reshape(-1, 3, 3)

        converter = vtk.vtkCellDataToPointData()
        converter.ProcessAllArraysOn()
        converter.SetInputConnection(cell_derivatives.GetOutputPort())
        converter.Update()
        vector_gradient = vtk_to_numpy(converter.GetOutput().GetPointData().GetArray("VectorGradient")).reshape(-1, 3, 3)

        return vector_gradient

    def compute_strain_rate(
            self
    ) -> np.ndarray:
        """
        Compute the strain rate tensor.
        """
        grad_u = self.compute_vector_derivatives(field="Vitesse")
        strain_rate = 2*abs(grad_u + grad_u.transpose(0, 2, 1))
        strain_rate = np.array([np.sum(strain_rate[i,:,:]) for i in range(len(strain_rate))])
        return strain_rate
    
    def compute_viscous_dissipation(
            self
    ) -> np.ndarray:
        """
        Compute the viscous dissipation.
        """
        strain_rate = self.compute_strain_rate()
        mu = self.vtu_data.GetOutput().GetPointData().GetArray("mu")
        mu = vtk_to_numpy(mu)
        viscous_dissipation = mu*strain_rate
        return viscous_dissipation