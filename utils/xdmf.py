
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

    def get_points(self, displacement: np.ndarray, displacement0: np.ndarray) -> np.ndarray:
        """
        Get the points of the mesh.
        """
        points =  vtk_to_numpy(self.xdmf_data.GetOutput().GetBlock(0).GetPoints().GetData())
        if displacement is not None and displacement0 is not None:
            points += displacement - displacement0
        return points

    def get_cells(self) -> np.ndarray:
        """
        Get the cells of the mesh.
        """
        nCells = self.xdmf_data.GetOutput().GetBlock(0).GetNumberOfCells()
        cells = np.zeros((nCells, 4), dtype=int)
        for i in range(nCells):
            cell = self.xdmf_data.GetOutput().GetBlock(0).GetCell(i)
            for j in range(4):
                cells[i,j] = cell.GetPointId(j)
        return cells

    def get_time_steps(self):
        """
        Get the number of time steps.
        """
        return self.xdmf_data.GetOutputInformation(0).Get(vtk.vtkCompositeDataPipeline.TIME_STEPS())
    
    def get_field_list(self) -> list[str]:
        """
        Get the list of fields.
        """
        return [self.xdmf_data.GetOutput().GetBlock(0).GetPointData().GetArrayName(i) for i in range(self.xdmf_data.GetOutput().GetBlock(0).GetPointData().GetNumberOfArrays())]
    
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
        strain_rate = np.sqrt(2*grad_u[:,0,0]**2 + 2*grad_u[:,1,1]**2 + 2*grad_u[:,2,2]**2 + (grad_u[:,0,1] + grad_u[:,1,0])**2 + (grad_u[:,0,2] + grad_u[:,2,0])**2 + (grad_u[:,1,2] + grad_u[:,2,1])**2)
        return strain_rate
    
    def compute_viscous_dissipation(
            self
    ) -> np.ndarray:
        """
        Compute the viscous dissipation.
        """
        strain_rate = self.compute_strain_rate()
        mu = self.xdmf_data.GetOutput().GetBlock(0).GetPointData().GetArray("mu")
        # If mu is not none
        if mu:
            mu = vtk_to_numpy(mu)
            viscous_dissipation = mu*strain_rate
            return viscous_dissipation
        # If mu is none
        mu = 0.004*np.ones(len(strain_rate))
        viscous_dissipation = mu*strain_rate/2
        return viscous_dissipation