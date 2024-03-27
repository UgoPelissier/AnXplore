import os
import meshio
import numpy as np

from utils.xdmf import XDMF_Wrapper
from utils.points import min_max, mean_std, split_vessels_aneurysm, extract_surface_points, WSS_regions
from utils.triangles import surface_area, extract_surface_cells, aneurysm_cells, WSS_regions_cells, positive_v_y_cells
from utils.tetra import clip_tetra, volume_region
from utils.integrate import integrate_field_surface, integrate_field_volume

def compute_redundant_first(
        xdmf_file: XDMF_Wrapper,
        displacement: np.ndarray,
        displacement0: np.ndarray,
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> tuple:
    """
    Compute the first redundant of the indicators.
    """
    points = xdmf_file.get_points(displacement, displacement0)
    cells = xdmf_file.get_cells()
    WSS = xdmf_file.get_point_field('WSS')
    OSI = xdmf_file.get_point_field('OSI')
    TAWSS = xdmf_file.get_point_field('TAWSS')
    Vitesse = xdmf_file.get_point_field('Vitesse')

    vessels, aneurysm, aneurysm_indicator = split_vessels_aneurysm(points, orifice_origin, orifice_plane)
    aneurysm_surface_points = extract_surface_points(WSS, aneurysm)
    vessels_surface_points = extract_surface_points(WSS, vessels)
    aneurysm_tetra, vessels_tetra = clip_tetra(points, cells, orifice_origin, orifice_plane)

    surface_cells = extract_surface_cells(cells, WSS)
    aneurysm_surface_cells, vessels_surface_cells = aneurysm_cells(surface_cells, aneurysm_indicator)
    aneurysm_area = surface_area(points, aneurysm_surface_cells)
    vessels_area = surface_area(points, vessels_surface_cells)
    aneurysm_volume = volume_region(points, aneurysm_tetra)
    vessels_volume = volume_region(points, vessels_tetra)
    return points, cells, WSS, OSI, TAWSS, Vitesse, vessels, aneurysm, aneurysm_surface_points, vessels_surface_points, surface_cells, aneurysm_surface_cells, vessels_surface_cells, aneurysm_area, vessels_area, aneurysm_tetra, vessels_tetra, aneurysm_volume, vessels_volume

def compute_redundant_second(
        WSS: np.ndarray,
        aneurysm_surface_cells: np.ndarray,
        mean_wss_vessels: float,
        std_wss_vessels: float
) -> tuple:
    """
    Compute the second redundant of the indicators.
    """
    regions = WSS_regions(WSS, mean_wss_vessels, std_wss_vessels)
    WSS_low_cells, WSS_high_cells = WSS_regions_cells(aneurysm_surface_cells, regions)
    return WSS_low_cells, WSS_high_cells

def mean_velocity(
        Vitesse: np.ndarray,
        region: np.ndarray
) -> float:
    """
    Compute the mean velocity in a region.
    """
    return np.mean(np.linalg.norm(Vitesse[region], axis=1))
        
def wss(
        points: np.ndarray,
        WSS: np.ndarray,
        surface_cells: np.ndarray,
        area: float,
        surface_points: np.ndarray
) -> tuple[float]:
    """
    Compute the min, max, mean and std WSS values in the aneurysm.
    """
    mean_wss = integrate_field_surface(surface_cells, points, WSS)/area
    min_wss, max_wss = min_max(WSS[surface_points])
    mean_wss_pw, std_wss_pw = mean_std(WSS[surface_points])

    return mean_wss, min_wss, max_wss, mean_wss_pw, std_wss_pw

def osi(
        points,
        OSI: np.ndarray,
        surface_cells: np.ndarray,
        area: float,
        surface_points: np.ndarray
) -> tuple[float]:
    """
    Compute the min, max, mean and std OSI values in the aneurysm.
    """
    mean_osi_aneurysm = integrate_field_surface(surface_cells, points, OSI)/area
    min_osi_aneurysm, max_osi_aneurysm = min_max(OSI[surface_points])
    mean_osi_pw, std_osi_pw = mean_std(OSI[surface_points])

    return mean_osi_aneurysm, min_osi_aneurysm, max_osi_aneurysm, mean_osi_pw, std_osi_pw

def tawss(
        points,
        TAWSS: np.ndarray,
        surface_cells: np.ndarray,
        area: float,
        surface_points: np.ndarray
) -> tuple[float]:
    """
    Compute the min, max, mean and std TAWSS values in the aneurysm.
    """
    mean_tawss_aneurysm = integrate_field_surface(surface_cells, points, TAWSS)/area
    min_tawss_aneurysm, max_tawss_aneurysm = min_max(TAWSS[surface_points])
    mean_tawss_pw, std_tawss_pw = mean_std(TAWSS[surface_points])

    return mean_tawss_aneurysm, min_tawss_aneurysm, max_tawss_aneurysm, mean_tawss_pw, std_tawss_pw

def compute_KER(
        points: np.ndarray,
        Vitesse: np.ndarray,
        aneurysm_tetra: np.ndarray,
        vessels_tetra: np.ndarray,
        aneurysm_volume: float,
        vessels_volume: float
) -> float:
    """
    Compute the Kinetic Energy Ratio
    """
    V = np.linalg.norm(Vitesse, axis=1)**2/2
    k_a = integrate_field_volume(aneurysm_tetra, points, V)
    k_v = integrate_field_volume(vessels_tetra, points, V)
    KER = (k_a/aneurysm_volume)/(k_v/vessels_volume)
    return KER

def compute_VDR(
        xdmf_file: XDMF_Wrapper,
        points: np.ndarray,
        aneurysm_tetra: np.ndarray,
        vessels_tetra: np.ndarray,
        aneurysm_volume: float,
        vessels_volume: float
) -> float:
    """
    Compute the Viscous Dissipation Ratio
    """
    viscous_dissipation = xdmf_file.compute_viscous_dissipation()
    phi_a = integrate_field_volume(aneurysm_tetra, points, viscous_dissipation)
    phi_v = integrate_field_volume(vessels_tetra, points, viscous_dissipation)
    VDR = (phi_a/aneurysm_volume)/(phi_v/vessels_volume)
    return VDR

def compute_LSA(
        points: np.ndarray,
        aneurysm_area: float,
        WSS_low_cells: np.ndarray,
        WSS_high_cells: np.ndarray
) -> tuple[float]:
    """
    Compute the Low Shear Area and High Shear Area
    """
    WSS_low_area = surface_area(points, WSS_low_cells)
    WSS_high_area = surface_area(points, WSS_high_cells)
    LSA = WSS_low_area/aneurysm_area
    HSA = WSS_high_area/aneurysm_area
    return LSA, HSA

def compute_SCI(
        points: np.ndarray,
        WSS: np.ndarray,
        aneurysm_surface_cells: np.ndarray,
        WSS_high_cells: np.ndarray,
        HSA: float
) -> float:
    """
    Compute the integrals of the WSS in the aneurysm and vessels
    """
    F_a = integrate_field_surface(aneurysm_surface_cells, points, WSS)
    F_h = integrate_field_surface(WSS_high_cells, points, WSS)
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
        xdmf_file_annex: XDMF_Wrapper,
        t: float,
        displacement0: np.ndarray,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float]
) -> list[float]:
    """
    Compute the indicators over a cardiac cycle.
    """
    if xdmf_file_annex is not None:
        displacement = xdmf_file_annex.get_point_field('DisplacementF')
    else:
        displacement = None
    # Compute first redundants
    points, cells, WSS, OSI, TAWSS, Vitesse, vessels, aneurysm, aneurysm_surface_points, vessels_surface_points, surface_cells, aneurysm_surface_cells, vessels_surface_cells, aneurysm_area, vessels_area, aneurysm_tetra, vessels_tetra, aneurysm_volume, vessels_volume = compute_redundant_first(xdmf_file, displacement, displacement0, orifice_origin, orifice_plane)

    # Compute first indicators
    mean_velocity_aneurysm = mean_velocity(Vitesse, aneurysm)
    mean_wss_aneurysm, min_wss_aneurysm, max_wss_aneurysm, mean_wss_aneurysm_pw, std_wss_aneurysm_pw = wss(points, WSS, aneurysm_surface_cells, aneurysm_area, aneurysm_surface_points)
    mean_wss_vessels, min_wss_vessels, max_wss_vessels, mean_wss_vessels_pw, std_wss_vessels_pw = wss(points, WSS, vessels_surface_cells, vessels_area, vessels_surface_points)
    mean_osi_aneurysm, min_osi_aneurysm, max_osi_aneurysm, mean_osi_aneurysm_pw, std_osi_aneurysm_pw = osi(points, OSI, aneurysm_surface_cells, aneurysm_area, aneurysm_surface_points)
    mean_tawss_aneurysm, min_tawss_aneurysm, max_tawss_aneurysm, mean_tawss_aneurysm_pw, std_tawss_aneurysm_pw = tawss(points, TAWSS, aneurysm_surface_cells, aneurysm_area, aneurysm_surface_points)
    KER = compute_KER(points, Vitesse, aneurysm_tetra, vessels_tetra, aneurysm_volume, vessels_volume)
    VDR = compute_VDR(xdmf_file, points, aneurysm_tetra, vessels_tetra, aneurysm_volume, vessels_volume)
    ICI = compute_ICI(xdmf_file, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane)
    
    # Compute second redundants
    WSS_low_cells, WSS_high_cells = compute_redundant_second(WSS, aneurysm_surface_cells, mean_wss_vessels_pw, std_wss_vessels_pw)
    
    # Compute second indicators
    LSA, HSA = compute_LSA(points, aneurysm_area, WSS_low_cells, WSS_high_cells)
    SCI = compute_SCI(points, WSS, aneurysm_surface_cells, WSS_high_cells, HSA)

    return [t, mean_velocity_aneurysm, mean_wss_aneurysm, min_wss_aneurysm, max_wss_aneurysm, mean_wss_aneurysm_pw, std_wss_aneurysm_pw, mean_osi_aneurysm, min_osi_aneurysm, max_osi_aneurysm, mean_osi_aneurysm_pw, std_osi_aneurysm_pw, mean_tawss_aneurysm, min_tawss_aneurysm, max_tawss_aneurysm, mean_tawss_aneurysm_pw, std_tawss_aneurysm_pw, KER, VDR, LSA, HSA, SCI, ICI]