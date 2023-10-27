import os
import os.path as osp
import meshio
import numpy as np

from utils.vtk import VTU_Wrapper
from utils.points import min_max, mean_std, split_vessels_aneurysm, extract_surface_points, WSS_regions
from utils.triangles import surface_area, extract_surface_cells, aneurysm_cells, WSS_regions_cells, positive_v_y_cells
from utils.tetra import slice_tetra, volume_region
from utils.integrate import integrate_field_surface, integrate_field_volume

def wss(
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> tuple[float]:
    # Split and extract surface points and print the minimum and maximum WSS values
    vessels, aneurysm, _ = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    surface_points = extract_surface_points(mesh.point_data['WSS'], np.array(range(len(mesh.points))))
    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    vessels_surface_points = extract_surface_points(mesh.point_data['WSS'], vessels)
    min_wss, max_wss = min_max(mesh.point_data['WSS'][surface_points])
    min_wss_aneurysm, max_wss_aneurysm = min_max(mesh.point_data['WSS'][aneurysm_surface_points])
    min_wss_vessels, max_wss_vessels = min_max(mesh.point_data['WSS'][vessels_surface_points])

    # Mean and std of WSS in the vessels
    mean_wss, std_wss = mean_std(mesh.point_data['WSS'][surface_points])
    mean_wss_aneurysm, std_wss_aneurysm = mean_std(mesh.point_data['WSS'][aneurysm_surface_points])
    mean_wss_vessels, std_wss_vessels = mean_std(mesh.point_data['WSS'][vessels_surface_points])

    return min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels

def osi(
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> tuple[float]:
    _, aneurysm, _ = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    min_osi_aneurysm, max_osi_aneurysm = min_max(mesh.point_data['OSI'][aneurysm_surface_points])
    mean_osi_aneurysm, std_osi_aneurysm = mean_std(mesh.point_data['OSI'][aneurysm_surface_points])
    return min_osi_aneurysm, max_osi_aneurysm, mean_osi_aneurysm, std_osi_aneurysm

def compute_KER(
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> float:
    # Split the mesh into two volumes: aneurysm and vessels
    aneurysm_tetra, vessels_tetra, _ = slice_tetra(mesh, orifice_origin, orifice_plane)

    # Compute the Kinetic Energy Ration and the Viscous Dissipation Ratio
    k_a = integrate_field_volume(aneurysm_tetra, mesh.points, np.linalg.norm(mesh.point_data['Vitesse'], axis=1)**2)
    V_a = volume_region(mesh.points, aneurysm_tetra)
    k_v = integrate_field_volume(vessels_tetra, mesh.points, np.linalg.norm(mesh.point_data['Vitesse'], axis=1)**2)
    V_v = volume_region(mesh.points, vessels_tetra)
    KER = (k_a/V_a)/(k_v/V_v)

    return KER

def compute_VDR(
        vtu_dir: str,
        filename: str,
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> float:
    aneurysm_tetra, vessels_tetra, _ = slice_tetra(mesh, orifice_origin, orifice_plane)

    vtu_file = VTU_Wrapper(osp.join(vtu_dir, filename))
    viscous_dissipation = vtu_file.compute_viscous_dissipation()

    phi_a = integrate_field_volume(aneurysm_tetra, mesh.points, viscous_dissipation)
    phi_v = integrate_field_volume(vessels_tetra, mesh.points, viscous_dissipation)

    V_a = volume_region(mesh.points, aneurysm_tetra)
    V_v = volume_region(mesh.points, vessels_tetra)

    VDR = (phi_a/V_a)/(phi_v/V_v)

    return VDR

def compute_LSA(
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> tuple[float]:
    _, aneurysm, aneurysm_indicator = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    _, _, _, _, _, _, _, _, _, _, mean_wss_vessels, std_wss_vessels = wss(mesh, orifice_origin, orifice_plane)

    # Extract the surface cells
    surface_cells = extract_surface_cells(mesh)

    # WSS regions (point data)
    _, _, regions = WSS_regions(aneurysm_surface_points, mesh.point_data['WSS'], mean_wss_vessels, std_wss_vessels)

    # Compute area of the vessels and aneurysm
    aneurysm_surface_cells, _ = aneurysm_cells(surface_cells, aneurysm_indicator)
    aneurysm_area = surface_area(mesh.points, aneurysm_surface_cells)
    WSS_low_cells, WSS_high_cells = WSS_regions_cells(surface_cells, regions)
    WSS_low_area = surface_area(mesh.points, WSS_low_cells)
    WSS_high_area = surface_area(mesh.points, WSS_high_cells)

    LSA = WSS_low_area/aneurysm_area
    HSA = WSS_high_area/aneurysm_area

    return LSA, HSA

def compute_SCI(
        mesh: meshio.Mesh,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> float:
    """
    Compute the integrals of the WSS in the aneurysm and vessels
    """
    _, aneurysm, aneurysm_indicator = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    surface_cells = extract_surface_cells(mesh)
    aneurysm_surface_cells, _ = aneurysm_cells(surface_cells, aneurysm_indicator)

    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    _, _, _, _, _, _, _, _, _, _, mean_wss_vessels, std_wss_vessels = wss(mesh, orifice_origin, orifice_plane)
    _, _, regions = WSS_regions(aneurysm_surface_points, mesh.point_data['WSS'], mean_wss_vessels, std_wss_vessels)
    _, WSS_high_cells = WSS_regions_cells(surface_cells, regions)

    _, HSA = compute_LSA(mesh, orifice_origin, orifice_plane)

    F_a = integrate_field_surface(aneurysm_surface_cells, mesh.points, mesh.point_data['WSS'])    
    F_h = integrate_field_surface(WSS_high_cells, mesh.points, mesh.point_data['WSS'])
    SCI = (F_h/F_a)/HSA
    return SCI

def compute_ICI(
        vtu_dir: str,
        filename: str,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> float: 

    vtu_file = VTU_Wrapper(osp.join(vtu_dir, filename))

    vessel_in_out, _ = vtu_file.get_slice_data(vessel_in_out_origin, vessel_in_out_plane, "Vitesse", vtu_dir, f"{filename[:-4]}_vessel_in_out.vtu")
    pos_v_y_vessel_in_out_cells = positive_v_y_cells(vessel_in_out)
    Q_v = integrate_field_surface(pos_v_y_vessel_in_out_cells, vessel_in_out.points, vessel_in_out.point_data['Vitesse'][:,1])

    orifice, _ = vtu_file.get_slice_data(orifice_origin, orifice_plane, "Vitesse", vtu_dir, f"{filename[:-4]}_orifice.vtu")
    pos_v_y_orifice_cells = positive_v_y_cells(orifice)
    Q_i = integrate_field_surface(pos_v_y_orifice_cells, orifice.points, orifice.point_data['Vitesse'][:,1])

    A_i = surface_area(orifice.points, pos_v_y_orifice_cells)
    A_o = surface_area(orifice.points, orifice.cells[0].data)

    ICI = (Q_i/Q_v)/(A_i/A_o)

    return ICI

def compute_indicators(
        vtu_dir: str,
        filename: str,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> list[float]:
    """
    Compute the indicators over a cardiac cycle.
    """
    # Load mesh
    mesh = meshio.read(os.path.join(vtu_dir, filename))

    # Compute indicators
    min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels = wss(mesh, orifice_origin, orifice_plane)
    _, max_osi_aneurysm, mean_osi_aneurysm, _ = osi(mesh, orifice_origin, orifice_plane)
    KER = compute_KER(mesh, orifice_origin, orifice_plane)
    VDR = compute_VDR(vtu_dir, filename, mesh, orifice_origin, orifice_plane)
    LSA, HSA = compute_LSA(mesh, orifice_origin, orifice_plane)
    SCI = compute_SCI(mesh, orifice_origin, orifice_plane)
    ICI = compute_ICI(vtu_dir, filename, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane)

    return [min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels, mean_osi_aneurysm, max_osi_aneurysm, KER, VDR, LSA, HSA, SCI, ICI]