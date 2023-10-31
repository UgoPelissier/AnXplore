import glob
import os
import os.path as osp
import numpy as np
from alive_progress import alive_bar

from utils.xdmf import XDMF_Wrapper
from utils.indicators import compute_indicators

if __name__ == '__main__':

    data_dir = "data"
    cases = ['fsi']

    vessel_in_out_origin = [0.0, 0.001, 0.0]
    vessel_in_out_plane = [0.0, 1.0, 0.0]

    orifice_origin = [0.0, 5.3, 0.0]
    orifice_plane = [0.0, 1.0, 0.0]

    T_cardiac_cycle = 20

    for case in cases:
        filenames = glob.glob(osp.join(data_dir, case, "*.xdmf"))

        with alive_bar(T_cardiac_cycle, title=f"Computing indicators for {case} case") as bar:
            for filename in filenames:
                xdmf_file = XDMF_Wrapper(osp.join(filename))    

                os.makedirs(osp.join(data_dir, "csv"), exist_ok=True)
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
                    fname=osp.join(data_dir, "csv", f"{filename.split('/')[-1][:-5]}.csv"),
                    X=np.array(indicators),
                    fmt="%.5f",
                    header="min_wss max_wss min_wss_aneurysm max_wss_aneurysm min_wss_vessels max_wss_vessels mean_wss std_wss mean_wss_aneurysm std_wss_aneurysm mean_wss_vessels std_wss_vessels mean_osi_aneurysm max_osi_aneurysm KER VDR LSA HSA SCI ICI",
                    comments='')
                
                bar()
        print("Done.")