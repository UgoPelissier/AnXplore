from utils.xdmf import XDMF_Wrapper
import numpy as np
import glob
from alive_progress import alive_bar
import os.path as osp

def drop_off_altitude(file_path: str) -> float:
    xdmf_file = XDMF_Wrapper(file_path)
    v_y = xdmf_file.get_point_field("Vitesse")[:,1]

    # points index where 0.5<x<2, 5.25<y<6.25, -0.5<z<0.5
    points = xdmf_file.get_points()
    point_idx = (points[:,0]>0.5) & (points[:,0]<2) & (points[:,1]>5.25) & (points[:,1]<6.25) & (points[:,2]>-0.5) & (points[:,2]<0.5)

    # cells containing at least one of the points
    cells = xdmf_file.get_cells()
    cells_idx = np.zeros(cells.shape[0], dtype=bool)
    for (i, cell) in enumerate(cells):
        if (point_idx[cell].sum()>3):
            neg = 0
            pos = 0
            for point in cell:
                if (v_y[point]<0):
                    neg += 1
                else:
                    pos += 1
            if (neg>0 and pos>0):
                cells_idx[i] = True

    # mean x and y position of the cells
    cells_pos = np.zeros((cells_idx.sum(), 2))
    for (i, cell) in enumerate(cells[cells_idx]):
        cells_pos[i,0] = points[cell,0].mean()
        cells_pos[i,1] = points[cell,1].mean()
    return(cells_pos.max(axis=0)[1])

if __name__ == '__main__':
    data_dir = "data"
    cases = ['rigid', 'fsi']

    for case in cases:
        filenames = glob.glob(osp.join(data_dir, case, "*.xdmf"))
        with alive_bar(len(filenames), title=f"Computing drop-off for {case} case") as bar:
            with open(f'res/drop_off_{case}.csv', 'w') as f:
                for filename in filenames:
                    f.write(f'{filename.split("/")[-1].split(".")[0]}, {drop_off_altitude(filename)}\n')
                    bar()