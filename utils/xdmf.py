
from typing import Union, List
import numpy as np
import meshio
import vtk
from vtk.util.numpy_support import vtk_to_numpy #type: ignore
from matplotlib import tri

class XDMF_Wrapper(object):
    """
    Wrapper for the vtkXMLUnstructuredGridReader class.
    """

    def __init__(self, filename: str) -> None:
        xdmf_data = vtk.vtkXdmfReader()
        xdmf_data.SetFileName(filename)
        xdmf_data.Update()

        self.filename = filename
        self.xdmf_data = xdmf_data
        self.tetra_id = 10

    def get_points(self) -> np.ndarray:
        """
        Get the points of the mesh.
        """
        return vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetPoints().GetData())

    def get_cells(self) -> np.ndarray:
        """
        Get the cells of the mesh.
        """
        data = vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetCells().GetData())
        offsets = vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetCellLocationsArray())
        types = vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetCellTypesArray())

        cells = (data[:4*types.shape[0]]).reshape(-1,4)
        return cells

    def get_time_steps(self):
        """
        Get the number of time steps.
        """
        return self.xdmf_data.GetOutputInformation(0).Get(vtk.vtkCompositeDataPipeline.TIME_STEPS())
    
    def get_point_field(self, field: str) -> np.ndarray:
        """
        Get the field of the points.
        """
        return vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetPointData().GetArray(field))

    def update_time_step(self, time_step: float) -> None:
        """
        Update the time step.
        """
        self.xdmf_data.UpdateTimeStep(time_step)

    def get_slice_data(
        self,
        center: Union[List[float], np.ndarray],
        normal: Union[List[float], np.ndarray],
        field: str,
    ) -> tuple[meshio.Mesh, float]:
        """
        Get slice of data from using a normal and an center for the slice.
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(*center)
        plane.SetNormal(*normal)

        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(self.xdmf_data.GetOutput().GetBlock(0))
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

        return surface, cut_area
    
    def compute_vector_derivatives(
            self,
            field: str
    ) -> np.ndarray:
        """
        Compute the derivatives of a vector field.
        """
        cell_derivatives = vtk.vtkCellDerivatives()
        cell_derivatives.SetInputData(self.xdmf_data.GetOutput().GetBlock(0))
        cell_derivatives.SetTensorModeToComputeGradient()
        cell_derivatives.SetVectorModeToPassVectors()
        self.xdmf_data.GetOutput().GetBlock(0).GetPointData().SetActiveVectors(field)
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
        mu = self.xdmf_data.GetOutput().GetBlock(0).GetPointData().GetArray("mu")
        mu = vtk_to_numpy(mu)
        viscous_dissipation = mu*strain_rate
        return viscous_dissipation