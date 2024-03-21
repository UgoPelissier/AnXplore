import glob
import os
import os.path as osp
import numpy as np
from alive_progress import alive_bar
from multiprocessing import Pool

from utils.xdmf import XDMF_Wrapper
from utils.indicators import compute_indicators

def process(
        data_dir: str,
        res_dir: str,
        case: str,
        id: int,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float],
        T_cardiac_cycle: int
) -> None:
    """
    Compute the indicators for a given XDMF file.
    """
    print (f"Processing case {case} - id {id}")
    xdmf_file = XDMF_Wrapper(osp.join(data_dir, case, f"Resultats_MESH_{id}", "AllFields.xdmf"))
    indicators = []

    try:
        xdmf_file_annex = XDMF_Wrapper(osp.join(data_dir, case, f"Resultats_MESH_{id}", "ConvectiveVelocity.xdmf"))
        xdmf_file_annex.update_time_step(0)
        displacement0 = xdmf_file_annex.get_point_field('DisplacementF')
    except:
        xdmf_file_annex = None
        displacement0 = None

    # Compute indicators over a cardiac cycle
    time_steps = xdmf_file.get_time_steps()
    for t in time_steps:
        print (f"Processing - id {id} - time step {t}")
        xdmf_file.update_time_step(t)
        if xdmf_file_annex is not None:
            xdmf_file_annex.update_time_step(10*t)
        indicators.append(compute_indicators(xdmf_file, xdmf_file_annex, displacement0, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane))

    # Compute min, max and mean over the cardiac cycle
    indicators = np.array(indicators)
    min = np.min(indicators, axis=0)
    max = np.max(indicators, axis=0)
    mean = np.mean(indicators, axis=0)
    std = np.std(indicators, axis=0)
    indicators = np.concatenate((indicators, min.reshape((1,-1)), max.reshape((1,-1)), mean.reshape((1,-1)), std.reshape((1,-1))))

    np.savetxt(
        fname=osp.join(res_dir, case, f"{id}.csv"),
        X=np.array(indicators),
        fmt="%.10f",
        header="min_wss max_wss min_wss_aneurysm max_wss_aneurysm min_wss_vessels max_wss_vessels mean_wss std_wss mean_wss_aneurysm std_wss_aneurysm mean_wss_vessels std_wss_vessels min_osi_aneurysm max_osi_aneurysm mean_osi_aneurysm std_osi_aneurysm min_tawss_aneurysm max_tawss_aneurysm mean_tawss_aneurysm std_tawss_aneurysm KER VDR LSA HSA SCI ICI",
        comments=''
    )
    print(f"Done processing case {case} - id {id}")

def process_case(args):
    id, data_dir, res_dir, case, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane, T_cardiac_cycle = args
    process(data_dir, res_dir, case, id, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane, T_cardiac_cycle)

def main(parallel):
    data_dir = "/media/admin-upelissier/DATA"
    yy = ['78', '80', '82']
    yyv = [7.8, 8.0, 8.2]
    cases = ['rigid']
    start = 0
    end = 106
    exclude = 85
    if parallel:
        pool = Pool(processes=16)

    for y, yv in zip(yy, yyv):
        res_dir = osp.join("res", "csv", y)
        os.makedirs(res_dir, exist_ok=True)
        vessel_in_out_origin = [0.0, 0.001, 0.0]
        vessel_in_out_plane = [0.0, 1.0, 0.0]
        orifice_origin = [0.0, yv, 0.0]
        orifice_plane = [0.0, 1.0, 0.0]
        T_cardiac_cycle = 80

        for case in cases:
            os.makedirs(osp.join(res_dir, case), exist_ok=True)
            ids = [i for i in range(start, end) if i != exclude]

            if parallel:
                args_list = [(id, data_dir, res_dir, case, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane, T_cardiac_cycle) for id in ids]
                pool.map(process_case, args_list)
            else:
                with alive_bar(len(ids), title=f"Computing indicators for y={yv} - {case} case") as bar:
                    for id in ids:
                        process(data_dir, res_dir, case, id, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane, T_cardiac_cycle)
                        bar()
                print("Done.")

if __name__ == '__main__':
    parallel = True
    main(parallel)