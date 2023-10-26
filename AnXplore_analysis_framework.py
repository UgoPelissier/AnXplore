import os.path as osp
import meshio
import numpy as np

from utils.points import min_max, mean_std, split_vessels_aneurysm, extract_surface_points, WSS_regions
from utils.cells import surface_area, extract_surface_cells, aneurysm_cells, WSS_regions_cells, positive_v_y_cells, slice_tetra, volume_region
from utils.integrate import integrate_field_surface, integrate_field_volume
from utils.vtk import VTU_Wrapper

data_dir = "vtu/"
filename = "AnXplore178_FSI_00045.vtu"

vessel_in_out_origin = [0.0, 0.001, 0.0]
vessel_in_out_plane = [0.0, 1.0, 0.0]

orifice_origin = [0.0, 5.3, 0.0]
orifice_plane = [0.0, 1.0, 0.0]

if __name__ == '__main__':
    # Load the mesh
    path = osp.join(data_dir, filename)
    mesh = meshio.read(path)
    print(f"\nReading {filename}...\n{mesh}\n")

    # Split and extract surface points and print the minimum and maximum WSS values
    vessels, aneurysm, aneurysm_indicator = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    surface_points = extract_surface_points(mesh.point_data['WSS'], np.array(range(len(mesh.points))))
    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    vessels_surface_points = extract_surface_points(mesh.point_data['WSS'], vessels)
    min_wss, max_wss = min_max(mesh.point_data['WSS'][surface_points])
    min_wss_aneurysm, max_wss_aneurysm = min_max(mesh.point_data['WSS'][aneurysm_surface_points])
    min_wss_vessels, max_wss_vessels = min_max(mesh.point_data['WSS'][vessels_surface_points])

    print(f"min_wss = {min_wss}\nmax_wss = {max_wss}\n")
    print(f"min_wss_aneurysm = {min_wss_aneurysm}\nmax_wss_aneurysm = {max_wss_aneurysm}\n")
    print(f"min_wss_vessels = {min_wss_vessels}\nmax_wss_vessels = {max_wss_vessels}\n")

    # Mean and std of WSS in the vessels
    mean_wss, std_wss = mean_std(mesh.point_data['WSS'][surface_points])
    mean_wss_aneurysm, std_wss_aneurysm = mean_std(mesh.point_data['WSS'][aneurysm_surface_points])
    mean_wss_vessels, std_wss_vessels = mean_std(mesh.point_data['WSS'][vessels_surface_points])

    print(f"mean_wss = {mean_wss}\nstd_wss = {std_wss}\n")
    print(f"mean_wss_aneurysm = {mean_wss_aneurysm}\nstd_wss_aneurysm = {std_wss_aneurysm}\n")
    print(f"mean_wss_vessels = {mean_wss_vessels}\nstd_wss_vessels = {std_wss_vessels}\n")

    # Idem for OSI
    min_osi_aneurysm, max_osi_aneurysm = min_max(mesh.point_data['OSI'][aneurysm_surface_points])
    mean_osi_aneurysm, std_osi_aneurysm = mean_std(mesh.point_data['OSI'][aneurysm_surface_points])

    print(f"mean_osi_aneurysm = {mean_osi_aneurysm}\nmax_osi_aneurysm = {max_osi_aneurysm}\n")

    # Split the mesh into two volumes: aneurysm and vessels
    aneurysm_tetra, vessels_tetra, _ = slice_tetra(mesh, orifice_origin, orifice_plane)

    aneurysm_volume = meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", aneurysm_tetra)],
        point_data = {"Vitesse":  mesh.point_data['Vitesse'], "WSS": mesh.point_data['WSS'], "mu": mesh.point_data['mu']},
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_aneurysm.vtu"), aneurysm_volume)

    vessels_volume = meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", vessels_tetra)],
        point_data = {"Vitesse":  mesh.point_data['Vitesse'], "WSS": mesh.point_data['WSS'], "mu": mesh.point_data['mu']},
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_vessels.vtu"), vessels_volume)

    # Compute the Kinetic Energy Ration and the Viscous Dissipation Ratio
    k_a = integrate_field_volume(aneurysm_tetra, mesh.points, np.linalg.norm(mesh.point_data['Vitesse'], axis=1)**2)
    V_a = volume_region(mesh.points, aneurysm_tetra)
    k_v = integrate_field_volume(vessels_tetra, mesh.points, np.linalg.norm(mesh.point_data['Vitesse'], axis=1)**2)
    V_v = volume_region(mesh.points, vessels_tetra)
    KER = (k_a/V_a)/(k_v/V_v)
    print(f"k_a = {k_a}\nk_v = {k_v}\nV_a = {V_a}\nV_v = {V_v}\nKER = {KER:.2f}\n")

    vtu_file = VTU_Wrapper(path)
    viscous_dissipation = vtu_file.compute_viscous_dissipation()

    phi_a = integrate_field_volume(aneurysm_tetra, mesh.points, viscous_dissipation)
    phi_v = integrate_field_volume(vessels_tetra, mesh.points, viscous_dissipation)
    VDR = (phi_a/V_a)/(phi_v/V_v)
    print(f"phi_a = {phi_a}\nphi_v = {phi_v}\nVDR = {VDR:.2f}\n")

    # Extract the surface cells
    surface_cells = extract_surface_cells(mesh)

    # WSS regions (point data)
    WSS_low, WSS_high, regions = WSS_regions(aneurysm_surface_points, mesh.point_data['WSS'], mean_wss_vessels, std_wss_vessels)

    # Compute area of the vessels and aneurysm
    mesh_area = surface_area(mesh.points, surface_cells)
    aneurysm_surface_cells, vessels_surface_cells = aneurysm_cells(surface_cells, aneurysm_indicator)
    aneurysm_area = surface_area(mesh.points, aneurysm_surface_cells)
    vessels_area = surface_area(mesh.points, vessels_surface_cells)
    WSS_low_cells, WSS_high_cells = WSS_regions_cells(surface_cells, regions)
    WSS_low_area = surface_area(mesh.points, WSS_low_cells)
    WSS_high_area = surface_area(mesh.points, WSS_high_cells)

    LSA = WSS_low_area/aneurysm_area
    HSA = WSS_high_area/aneurysm_area

    print(f"mesh_area = {mesh_area}\naneurysm_area = {aneurysm_area}\nvessels_area = {vessels_area}\n")
    print(f"aneurysm_WSS_low_area = {WSS_low_area}\naneurysm_WSS_high_area = {WSS_high_area}\nLSA = {100*LSA:.1f}%\nHSA = {100*HSA:.1f}%\n")

    # Compute the integrals of the WSS in the aneurysm and vessels
    F_a = integrate_field_surface(aneurysm_surface_cells, mesh.points, mesh.point_data['WSS'])    
    F_h = integrate_field_surface(WSS_high_cells, mesh.points, mesh.point_data['WSS'])
    F_l = integrate_field_surface(WSS_low_cells, mesh.points, mesh.point_data['WSS'])
    SCI = (F_h/F_a)/HSA

    print(f"F_a = {F_a}\nF_h = {F_h}\nF_l = {F_l}\nSCI = {SCI:.2f}\n")

    vtu_file = VTU_Wrapper(path)

    if not osp.exists(osp.join(data_dir, f"{filename[:-4]}_vessel_in_out.vtu")):
        vessel_in_out, cut_area = vtu_file.get_slice_data(vessel_in_out_origin, vessel_in_out_plane, "Vitesse", data_dir, f"{filename[:-4]}_vessel_in_out.vtu")
    else:
        vessel_in_out = meshio.read(osp.join(data_dir, f"{filename[:-4]}_vessel_in_out.vtu"))
    pos_v_y_vessel_in_out_cells = positive_v_y_cells(vessel_in_out)
    Q_v = integrate_field_surface(pos_v_y_vessel_in_out_cells, vessel_in_out.points, vessel_in_out.point_data['Vitesse'][:,1])

    if not osp.exists(osp.join(data_dir, f"{filename[:-4]}_orifice.vtu")):
        orifice, cut_area = vtu_file.get_slice_data(orifice_origin, orifice_plane, "Vitesse", data_dir, f"{filename[:-4]}_orifice.vtu")
    else:
        orifice = meshio.read(osp.join(data_dir, f"{filename[:-4]}_orifice.vtu"))
    pos_v_y_orifice_cells = positive_v_y_cells(orifice)
    Q_i = integrate_field_surface(pos_v_y_orifice_cells, orifice.points, orifice.point_data['Vitesse'][:,1])

    A_i = surface_area(orifice.points, pos_v_y_orifice_cells)
    A_o = surface_area(orifice.points, orifice.cells[0].data)

    ICI = (Q_i/Q_v)/(A_i/A_o)

    print(f"Qi = {Q_i}\nQv = {Q_v}\nAi = {A_i}\nAo = {A_o}\nICI = {ICI:.2f}\n")

    # Save the results
    mesh_surface = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", surface_cells)],
        point_data = {"WSS":  mesh.point_data['WSS'], "aneurysm": aneurysm_indicator, "regions": regions}
    )
    meshio.write(osp.join(data_dir, f"{filename[:-4]}_surface.vtu"), mesh_surface)

    # Save indicators to csv file
    indicators = np.array([[min_wss, max_wss, min_wss_aneurysm, max_wss_aneurysm, min_wss_vessels, max_wss_vessels, mean_wss, std_wss, mean_wss_aneurysm, std_wss_aneurysm, mean_wss_vessels, std_wss_vessels, mean_osi_aneurysm, max_osi_aneurysm, KER, VDR, LSA, HSA, SCI, ICI]])
    np.savetxt(
        fname=osp.join(data_dir, f"{filename[:-4]}_indicators.csv"),
        X=indicators,
        fmt="%.5f",
        header="min_wss max_wss min_wss_aneurysm max_wss_aneurysm min_wss_vessels max_wss_vessels mean_wss std_wss mean_wss_aneurysm std_wss_aneurysm mean_wss_vessels std_wss_vessels mean_osi_aneurysm max_osi_aneurysm, KER, VDR, LSA, HSA, SCI, ICI",
        comments='')