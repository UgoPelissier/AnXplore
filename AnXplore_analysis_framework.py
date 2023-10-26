import os.path as osp
import meshio
import numpy as np
from alive_progress import alive_bar

from utils.points import min_max, mean_std, project_point_on_plane, split_vessels_aneurysm, extract_surface_points, WSS_regions
from utils.cells import surface_area, extract_surface_cells, aneurysm_cells, WSS_regions_cells, slice_triangles, positive_v_y_cells
from utils.integrate import integrate_field_surface
from utils.vtk import VTU_Wrapper

data_dir = "data/"
filename = "AnXplore178_FSI_00045.vtu"

vessel_in_out_origin = [0.0, 0.001, 0.0]
vessel_in_out_plane = [0.0, 1.0, 0.0]

orifice_origin = [0.0, 5.5, 0.0]
orifice_plane = [0.0, 1.0, 0.0]

if __name__ == '__main__':
    # Load the mesh
    path = osp.join(data_dir, filename)
    mesh = meshio.read(path)
    print(f"\nReading {filename}...\n{mesh}\n")

    # Extract the surface cells
    surface_cells = extract_surface_cells(mesh)

    # Split and extract surface points and print the minimum and maximum TAWSS values
    vessels, aneurysm, aneurysm_indicator = split_vessels_aneurysm(mesh, orifice_origin, orifice_plane)
    surface_points = extract_surface_points(mesh.point_data['WSS'], np.array(range(len(mesh.points))))
    aneurysm_surface_points = extract_surface_points(mesh.point_data['WSS'], aneurysm)
    vessels_surface_points = extract_surface_points(mesh.point_data['WSS'], vessels)
    min_wss, max_wss = min_max(mesh.point_data['WSS'][surface_points])
    min_wss_aneurysm, max_wss_aneurysm = min_max(mesh.point_data['WSS'][vessels_surface_points])
    min_wss_vessels, max_wss_vessels = min_max(mesh.point_data['WSS'][aneurysm_surface_points])

    print(f"min_wss = {min_wss}\nmax_wss = {max_wss}\n")
    print(f"min_wss_aneurysm = {min_wss_aneurysm}\nmax_wss_aneurysm = {max_wss_aneurysm}\n")
    print(f"min_wss_vessels = {min_wss_vessels}\nmax_wss_vessels = {max_wss_vessels}\n")

    # Mean and std of TAWSS in the vessels
    mean_wss, std_wss = mean_std(mesh.point_data['WSS'][surface_points])
    mean_wss_aneurysm, std_wss_aneurysm = mean_std(mesh.point_data['WSS'][aneurysm_surface_points])
    mean_wss_vessels, std_wss_vessels = mean_std(mesh.point_data['WSS'][vessels_surface_points])

    print(f"mean_wss = {mean_wss}\nstd_wss = {std_wss}\n")
    print(f"mean_wss_aneurysm = {mean_wss_aneurysm}\nstd_wss_aneurysm = {std_wss_aneurysm}\n")
    print(f"mean_wss_vessels = {mean_wss_vessels}\nstd_wss_vessels = {std_wss_vessels}\n")

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
    print(f"aneurysm_WSS_low_area = {WSS_low_area}\naneurysm_WSS_high_area = {WSS_high_area}\nLSA = {100*LSA:.1f}%\nHSA= {100*HSA:.1f}%\n")

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
    Q_v = np.linalg.norm(integrate_field_surface(pos_v_y_vessel_in_out_cells, vessel_in_out.points, vessel_in_out.point_data['Vitesse']))

    if not osp.exists(osp.join(data_dir, f"{filename[:-4]}_orifice.vtu")):
        orifice, cut_area = vtu_file.get_slice_data(orifice_origin, orifice_plane, "Vitesse", data_dir, f"{filename[:-4]}_orifice.vtu")
    else:
        orifice = meshio.read(osp.join(data_dir, f"{filename[:-4]}_orifice.vtu"))
    pos_v_y_orifice_cells = positive_v_y_cells(orifice)
    Q_i = np.linalg.norm(integrate_field_surface(pos_v_y_orifice_cells, orifice.points, orifice.point_data['Vitesse']))

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

    