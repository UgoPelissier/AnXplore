import os.path as osp
import glob
import pandas as pd
from alive_progress import alive_bar
import seaborn as sns

if __name__ == '__main__':

    data_dir = "data"
    cases = ['rigid', 'fsi']

    min_wss_aneurysm = []
    max_wss_aneurysm = []
    mean_wss_aneurysm = []
    min_osi_aneurysm = []
    max_osi_aneurysm = []
    mean_osi_aneurysm = []
    min_tawss_aneurysm = []
    max_tawss_aneurysm = []
    mean_tawss_aneurysm = []
    KER = []
    VDR = []
    LSA = []
    HSA = []
    SCI = []
    ICI = []

    for case in cases:
        filenames = glob.glob(osp.join(data_dir, "csv", case, "*.csv"))
        with alive_bar(len(filenames), title=f"Extracting indicators for {case} case") as bar:
            for filename in filenames:
                df = pd.read_csv(filename, sep=' ')
                min_wss_aneurysm += df['min_wss_aneurysm'].tolist()[:-4]
                max_wss_aneurysm += df['max_wss_aneurysm'].tolist()[:-4]
                mean_wss_aneurysm += df['mean_wss_aneurysm'].tolist()[:-4]
                min_osi_aneurysm += df['min_osi_aneurysm'].tolist()[-5:-4]
                max_osi_aneurysm += df['max_osi_aneurysm'].tolist()[-5:-4]
                mean_osi_aneurysm += df['mean_osi_aneurysm'].tolist()[-5:-4]
                min_tawss_aneurysm += df['min_tawss_aneurysm'].tolist()[-5:-4]
                max_tawss_aneurysm += df['max_tawss_aneurysm'].tolist()[-5:-4]
                mean_tawss_aneurysm += df['mean_tawss_aneurysm'].tolist()[-5:-4]
                KER += df['KER'].tolist()[:-4]
                VDR += df['VDR'].tolist()[:-4]
                LSA += df['LSA'].tolist()[:-4]
                HSA += df['HSA'].tolist()[:-4]
                SCI += df['SCI'].tolist()[:-4]
                ICI += df['ICI'].tolist()[:-4]
                bar()
        print("Done.")

    indicators = {
        'min_wss_aneurysm': min_wss_aneurysm,
        'max_wss_aneurysm': max_wss_aneurysm,
        'mean_wss_aneurysm': mean_wss_aneurysm,
        'min_osi_aneurysm': min_osi_aneurysm,
        'max_osi_aneurysm': max_osi_aneurysm,
        'mean_osi_aneurysm': mean_osi_aneurysm,
        'min_tawss_aneurysm': min_tawss_aneurysm,
        'max_tawss_aneurysm': max_tawss_aneurysm,
        'mean_tawss_aneurysm': mean_tawss_aneurysm,
        'KER': KER,
        'VDR': VDR,
        'LSA': LSA,
        'HSA': HSA,
        'SCI': SCI,
        'ICI': ICI,
    }

    # indicators = ['min_wss_aneurysm', 'max_wss_aneurysm', 'mean_wss_aneurysm', 'min_osi_aneurysm', 'max_osi_aneurysm', 'mean_osi_aneurysm', 'min_tawss_aneurysm', 'max_tawss_aneurysm', 'mean_tawss_aneurysm', 'KER', 'VDR', 'LSA', 'HSA', 'SCI', 'ICI']

    titles = ['$WSS_{min}$', '$WSS_{max}$', '$WSS_{avg}$', '$OSI_{min}$', '$OSI_{max}$', '$OSI_{avg}$', '$TAWSS_{min}$', '$TAWSS_{max}$', '$TAWSS_{avg}$', 'KER', 'VDR', 'LSA', 'HSA', 'SCI', 'ICI']

    for title, key in zip(titles, indicators):
        case = ['Rigid' for _ in range(int(len(indicators[key])/2))] + ['FSI' for _ in range(int(len(indicators[key])/2))]
        data = {
            key: indicators[key],
            'Case': case
        }
        df = pd.DataFrame(data=data)

        ax = sns.violinplot(data=df, y=key, hue="Case", linewidth=1.5, palette="Set2")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel(title)
        fig = ax.get_figure()
        fig.savefig(osp.join(data_dir, "violin", f'{key}.png'))
        ax.clear() # clear axes for next plot