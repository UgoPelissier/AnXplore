import os
import meshio
import numpy as np

from utils.xdmf import XDMF_Wrapper
from utils.points import min_max, mean_std, split_vessels_aneurysm, extract_surface_points, WSS_regions
from utils.triangles import surface_area, extract_surface_cells, aneurysm_cells, WSS_regions_cells, positive_v_y_cells
from utils.tetra import slice_tetra, volume_region
from utils.integrate import integrate_field_surface, integrate_field_volume

def compute_redundant_first(
        xdmf_file: XDMF_Wrapper,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> tuple:
    """
    Compute the first redundant of the indicators.
    """
    WSS = xdmf_file.get_point_field('WSS')
    vessels, aneurysm, aneurysm_indicator = split_vessels_aneurysm(xdmf_file.get_points(), orifice_origin, orifice_plane)
    aneurysm_surface_points = extract_surface_points(WSS, aneurysm)
    aneurysm_tetra, vessels_tetra = slice_tetra(xdmf_file.get_points(), xdmf_file.get_cells(), orifice_origin, orifice_plane)
    surface_cells = extract_surface_cells(xdmf_file.get_cells(), WSS)
    aneurysm_surface_cells, _ = aneurysm_cells(surface_cells, aneurysm_indicator)
    V_a = volume_region(xdmf_file.get_points(), aneurysm_tetra)
    V_v = volume_region(xdmf_file.get_points(), vessels_tetra)
    return vessels, aneurysm_surface_points, aneurysm_tetra, vessels_tetra, surface_cells, aneurysm_surface_cells, V_a, V_v

def compute_redundant_second(
        xdmf_file: XDMF_Wrapper,
        aneurysm_surface_points: np.ndarray,
        aneurysm_surface_cells: np.ndarray,
        mean_wss_vessels: float,
        std_wss_vessels: float
) -> tuple:
    """
    Compute the second redundant of the indicators.
    """
    regions = WSS_regions(xdmf_file.get_point_field('WSS'), mean_wss_vessels, std_wss_vessels)
    WSS_low_cells, WSS_high_cells = WSS_regions_cells(aneurysm_surface_cells, regions)
    return WSS_low_cells, WSS_high_cells

def wss(
        xdmf_file: XDMF_Wrapper,
        vessels: np.ndarray,
        aneurysm_surface_points: np.ndarray
) -> tuple[float]:
    """
    Compute the min, max, mean and std WSS values in the aneurysm and vessels."""
    WSS = xdmf_file.get_point_field('WSS')
    # Split and extract surface points and print the minimum and maximum WSS values
    surface_points = extract_surface_points(WSS, np.array(range(len(xdmf_file.get_points()))))
    vessels_surface_points = extract_surface_points(WSS, vessels)
    min_wss, max_wss = min_max(WSS[surface_points])
    min_wss_aneurysm, max_wss_aneurysm = min_max(WSS[aneurysm_surface_points])
    min_wss_vessels, max_wss_vessels = min_max(WSS[vessels_surface_points])

    # Mean and std of WSS in the vessels
    mean_wss, std_wss = mean_std(WSS[surface_points])
    mean_wss_aneurysm, std_wss_aneurysm = mean_std(WSS[aneurysm_surface_points])
    mean_wss_vessels, std_wss_vessels = mean_std(WSS[vessels_surface_points])

    return min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels

def osi(
        xdmf_file: XDMF_Wrapper,
        aneurysm_surface_points: np.ndarray
) -> tuple[float]:
    """
    Compute the min, max, mean and std OSI values in the aneurysm.
    """
    OSI = xdmf_file.get_point_field('OSI')
    min_osi_aneurysm, max_osi_aneurysm = min_max(OSI[aneurysm_surface_points])
    mean_osi_aneurysm, std_osi_aneurysm = mean_std(OSI[aneurysm_surface_points])
    return min_osi_aneurysm, max_osi_aneurysm, mean_osi_aneurysm, std_osi_aneurysm

def compute_KER(
        xdmf_file: XDMF_Wrapper,
        aneurysm_tetra: np.ndarray,
        vessels_tetra: np.ndarray,
        V_a: float,
        V_v: float
) -> float:
    """
    Compute the Kinetic Energy Ratio
    """
    V = np.linalg.norm(xdmf_file.get_point_field('Vitesse'), axis=1)**2
    k_a = integrate_field_volume(aneurysm_tetra, xdmf_file.get_points(), V)
    k_v = integrate_field_volume(vessels_tetra, xdmf_file.get_points(), V)
    KER = (k_a/V_a)/(k_v/V_v)
    return KER

def compute_VDR(
        xdmf_file: XDMF_Wrapper,
        aneurysm_tetra: np.ndarray,
        vessels_tetra: np.ndarray,
        V_a: float,
        V_v: float
) -> float:
    """
    Compute the Viscous Dissipation Ratio
    """
    viscous_dissipation = xdmf_file.compute_viscous_dissipation()
    phi_a = integrate_field_volume(aneurysm_tetra, xdmf_file.get_points(), viscous_dissipation)
    phi_v = integrate_field_volume(vessels_tetra, xdmf_file.get_points(), viscous_dissipation)
    return (phi_a/V_a)/(phi_v/V_v)

def compute_LSA(
        xdmf_file: XDMF_Wrapper,
        aneurysm_surface_cells: np.ndarray,
        WSS_low_cells: np.ndarray,
        WSS_high_cells: np.ndarray
) -> tuple[float]:
    """
    Compute the Low Shear Area and High Shear Area
    """
    aneurysm_area = surface_area(xdmf_file.get_points(), aneurysm_surface_cells)
    WSS_low_area = surface_area(xdmf_file.get_points(), WSS_low_cells)
    WSS_high_area = surface_area(xdmf_file.get_points(), WSS_high_cells)
    LSA = WSS_low_area/aneurysm_area
    HSA = WSS_high_area/aneurysm_area
    return LSA, HSA

def compute_SCI(
        xdmf_file: XDMF_Wrapper,
        aneurysm_surface_cells: np.ndarray,
        WSS_high_cells: np.ndarray,
        HSA: float
) -> float:
    """
    Compute the integrals of the WSS in the aneurysm and vessels
    """
    WSS = xdmf_file.get_point_field('WSS')
    F_a = integrate_field_surface(aneurysm_surface_cells, xdmf_file.get_points(), WSS)    
    F_h = integrate_field_surface(WSS_high_cells, xdmf_file.get_points(), WSS)
    SCI = (F_h/F_a)/HSA
    return SCI

def compute_ICI(
        xdmf_file: XDMF_Wrapper,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> float:
    """
    Compute the Inflow Concentration Index
    """
    vessel_in_out, _ = xdmf_file.get_slice_data(vessel_in_out_origin, vessel_in_out_plane, "Vitesse")
    pos_v_y_vessel_in_out_cells = positive_v_y_cells(vessel_in_out)
    Q_v = integrate_field_surface(pos_v_y_vessel_in_out_cells, vessel_in_out.points, vessel_in_out.point_data['Vitesse'][:,1])

    orifice, _ = xdmf_file.get_slice_data(orifice_origin, orifice_plane, "Vitesse")
    pos_v_y_orifice_cells = positive_v_y_cells(orifice)
    Q_i = integrate_field_surface(pos_v_y_orifice_cells, orifice.points, orifice.point_data['Vitesse'][:,1])

    A_i = surface_area(orifice.points, pos_v_y_orifice_cells)
    A_o = surface_area(orifice.points, orifice.cells[0].data)

    ICI = (Q_i/Q_v)/(A_i/A_o)

    return ICI

def compute_indicators(
        xdmf_file: XDMF_Wrapper,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> list[float]:
    """
    Compute the indicators over a cardiac cycle.
    """
    # Compute first redundants
    vessels, aneurysm_surface_points, aneurysm_tetra, vessels_tetra, surface_cells, aneurysm_surface_cells, V_a, V_v = compute_redundant_first(xdmf_file, orifice_origin, orifice_plane)

    # Compute first indicators
    min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels = wss(xdmf_file, vessels, aneurysm_surface_points)
    _, max_osi_aneurysm, mean_osi_aneurysm, _ = osi(xdmf_file, aneurysm_surface_points)
    KER = compute_KER(xdmf_file, aneurysm_tetra, vessels_tetra, V_a, V_v)
    VDR = compute_VDR(xdmf_file, aneurysm_tetra, vessels_tetra, V_a, V_v)
    ICI = compute_ICI(xdmf_file, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane)
    
    # Compute second redundants
    WSS_low_cells, WSS_high_cells = compute_redundant_second(xdmf_file, aneurysm_surface_points, aneurysm_surface_cells, mean_wss_vessels, std_wss_vessels)
    
    # Compute second indicators
    LSA, HSA = compute_LSA(xdmf_file, aneurysm_surface_cells, WSS_low_cells, WSS_high_cells)
    SCI = compute_SCI(xdmf_file, aneurysm_surface_cells, WSS_high_cells, HSA)

    return [min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels, mean_osi_aneurysm, max_osi_aneurysm, KER, VDR, LSA, HSA, SCI, ICI]