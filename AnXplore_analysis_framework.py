import os
import os.path as osp
import numpy as np
from alive_progress import alive_bar

from utils.vtk import decompress_h5
from utils.indicators import compute_indicators

if __name__ == '__main__':

    data_dir = "data"
    vtu_dir = osp.join(data_dir, "vtu")
    filename = "AnXplore178_FSI.xdmf"

    vessel_in_out_origin = [0.0, 0.001, 0.0]
    vessel_in_out_plane = [0.0, 1.0, 0.0]

    orifice_origin = [0.0, 5.3, 0.0]
    orifice_plane = [0.0, 1.0, 0.0]

    T_cardiac_cycle = 3

    decompress_h5(data_dir, filename)

    os.makedirs(osp.join(data_dir, "csv"), exist_ok=True)
    indicators = []
    cardiac_cycle = sorted(os.listdir(vtu_dir))[-T_cardiac_cycle:]
    with alive_bar(len(cardiac_cycle), title="Computing indicators over a cardiac cycle...") as bar:
        for time_step in cardiac_cycle:
            indicators.append(compute_indicators(vtu_dir, time_step, vessel_in_out_origin, vessel_in_out_plane, orifice_origin, orifice_plane))
            bar()

    # Compute min, max and mean over the cardiac cycle
    indicators = np.array(indicators)
    min = np.min(indicators, axis=0)
    max = np.max(indicators, axis=0)
    mean = np.mean(indicators, axis=0)
    std = np.std(indicators, axis=0)
    indicators = np.concatenate((indicators, min.reshape((1,-1)), max.reshape((1,-1)), mean.reshape((1,-1)), std.reshape((1,-1))))

    np.savetxt(
        fname=osp.join(data_dir, "csv", f"{filename[:-4]}_indicators.csv"),
        X=np.array(indicators),
        fmt="%.5f",
        header="min_wss max_wss min_wss_aneurysm max_wss_aneurysm min_wss_vessels max_wss_vessels mean_wss std_wss mean_wss_aneurysm std_wss_aneurysm mean_wss_vessels std_wss_vessels mean_osi_aneurysm max_osi_aneurysm, KER, VDR, LSA, HSA, SCI, ICI",
        comments='')
    
    os.system(f"rm -rf {vtu_dir}")