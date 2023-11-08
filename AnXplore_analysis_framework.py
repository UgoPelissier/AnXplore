import glob
import os
import os.path as osp
import numpy as np
from alive_progress import alive_bar

from utils.xdmf import XDMF_Wrapper
from utils.indicators import compute_indicators

def process(
        filename: str,
        vessel_in_out_origin: list[float],
        vessel_in_out_plane: list[float],
        orifice_origin: list[float],
        orifice_plane: list[float],
        T_cardiac_cycle: int
) -> None:
    """
    Compute the indicators for a given XDMF file.
    """
    xdmf_file = XDMF_Wrapper(osp.join(filename))  
    indicators = []

    # Compute indicators over a cardiac cycle
    time_steps = xdmf_file.get_time_steps()
    for t in time_steps[-T_cardiac_cycle:]:
        xdmf_file.update_time_step(t)
        indicators.append(compute_indicators(xdmf_file, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane))

    # Compute min, max and mean over the cardiac cycle
    indicators = np.array(indicators)
    min = np.min(indicators, axis=0)
    max = np.max(indicators, axis=0)
    mean = np.mean(indicators, axis=0)
    std = np.std(indicators, axis=0)
    indicators = np.concatenate((indicators, min.reshape((1,-1)), max.reshape((1,-1)), mean.reshape((1,-1)), std.reshape((1,-1))))

    np.savetxt(
        fname=osp.join(data_dir, "csv", case, f"{filename.split('/')[-1][:-5]}.csv"),
        X=np.array(indicators),
        fmt="%.5f",
        header="min_wss max_wss min_wss_aneurysm max_wss_aneurysm min_wss_vessels max_wss_vessels mean_wss std_wss mean_wss_aneurysm std_wss_aneurysm mean_wss_vessels std_wss_vessels min_osi_aneurysm max_osi_aneurysm mean_osi_aneurysm std_osi_aneurysm min_tawss_aneurysm max_tawss_aneurysm mean_tawss_aneurysm std_tawss_aneurysm KER VDR LSA HSA SCI ICI",
        comments=''
    )

if __name__ == '__main__':

    data_dir = "data"
    res_dir = "res"
    cases = ['rigid', 'fsi']

    os.makedirs(osp.join(res_dir, "csv"), exist_ok=True)

    vessel_in_out_origin = [0.0, 0.001, 0.0]
    vessel_in_out_plane = [0.0, 1.0, 0.0]

    orifice_origin = [0.0, 5.3, 0.0]
    orifice_plane = [0.0, 1.0, 0.0]

    T_cardiac_cycle = 20

    for case in cases:
        os.makedirs(osp.join(res_dir, "csv", case), exist_ok=True)
        filenames = glob.glob(osp.join(data_dir, case, "*.xdmf"))
        threads = []

        with alive_bar(len(filenames), title=f"Computing indicators for {case} case") as bar:
            for filename in filenames:
                process(filename, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane, T_cardiac_cycle)
                bar()
        print("Done.")