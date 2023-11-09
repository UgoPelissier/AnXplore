import os
import os.path as osp
import glob
import pandas as pd
from alive_progress import alive_bar
import seaborn as sns

if __name__ == '__main__':

    res_dir = "res"
    cases = ['rigid', 'fsi']

    os.makedirs(osp.join(res_dir), exist_ok=True)
    os.makedirs(osp.join(res_dir, "violin"), exist_ok=True)

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
        filenames = glob.glob(osp.join(res_dir, "csv", case, "*.csv"))
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
        fig.savefig(osp.join(res_dir, "violin", f'{key}.png'))
        ax.clear()

    min_wss_aneurysm = [[], [], []]
    max_wss_aneurysm = [[], [], []]
    mean_wss_aneurysm = [[], [], []]
    min_osi_aneurysm = [[], [], []]
    max_osi_aneurysm = [[], [], []]
    mean_osi_aneurysm = [[], [], []]
    KER = [[], [], []]
    VDR = [[], [], []]
    LSA = [[], [], []]
    HSA = [[], [], []]
    SCI = [[], [], []]
    ICI = [[], [], []]

    for case in cases:
        filenames = glob.glob(osp.join(res_dir, "csv", case, "*.csv"))
        for filename in filenames:
                df = pd.read_csv(filename, sep=' ')

                min_wss_aneurysm[0] += df['min_wss_aneurysm'].tolist()[:-4]
                max_wss_aneurysm[0] += df['max_wss_aneurysm'].tolist()[:-4]
                mean_wss_aneurysm[0] += df['mean_wss_aneurysm'].tolist()[:-4]
                min_osi_aneurysm[0] += df['min_osi_aneurysm'].tolist()[-5:-4]
                max_osi_aneurysm[0] += df['max_osi_aneurysm'].tolist()[-5:-4]
                mean_osi_aneurysm[0] += df['mean_osi_aneurysm'].tolist()[-5:-4]
                KER[0] += df['KER'].tolist()[:-4]
                VDR[0] += df['VDR'].tolist()[:-4]
                LSA[0] += df['LSA'].tolist()[:-4]
                HSA[0] += df['HSA'].tolist()[:-4]
                SCI[0] += df['SCI'].tolist()[:-4]
                ICI[0] += df['ICI'].tolist()[:-4]

                min_wss_aneurysm[1] +=  [case for _ in range(len(df['min_wss_aneurysm'].tolist()[:-4]))]
                max_wss_aneurysm[1] +=  [case for _ in range(len(df['max_wss_aneurysm'].tolist()[:-4]))]
                mean_wss_aneurysm[1] +=  [case for _ in range(len(df['mean_wss_aneurysm'].tolist()[:-4]))]
                min_osi_aneurysm[1] +=  [case for _ in range(len(df['min_osi_aneurysm'].tolist()[-5:-4]))]
                max_osi_aneurysm[1] +=  [case for _ in range(len(df['max_osi_aneurysm'].tolist()[-5:-4]))]
                mean_osi_aneurysm[1] +=  [case for _ in range(len(df['mean_osi_aneurysm'].tolist()[-5:-4]))]
                KER[1] +=  [case for _ in range(len(df['KER'].tolist()[:-4]))]
                VDR[1] +=  [case for _ in range(len(df['VDR'].tolist()[:-4]))]
                LSA[1] +=  [case for _ in range(len(df['LSA'].tolist()[:-4]))]
                HSA[1] +=  [case for _ in range(len(df['HSA'].tolist()[:-4]))]
                SCI[1] +=  [case for _ in range(len(df['SCI'].tolist()[:-4]))]
                ICI[1] +=  [case for _ in range(len(df['ICI'].tolist()[:-4]))]

                min_wss_aneurysm[2] +=  ['baseline' for _ in range(len(df['min_wss_aneurysm'].tolist()[:-4]))]
                max_wss_aneurysm[2] +=  ['baseline' for _ in range(len(df['max_wss_aneurysm'].tolist()[:-4]))]
                mean_wss_aneurysm[2] +=  ['baseline' for _ in range(len(df['mean_wss_aneurysm'].tolist()[:-4]))]
                min_osi_aneurysm[2] +=  ['baseline' for _ in range(len(df['min_osi_aneurysm'].tolist()[-5:-4]))]
                max_osi_aneurysm[2] +=  ['baseline' for _ in range(len(df['max_osi_aneurysm'].tolist()[-5:-4]))]
                mean_osi_aneurysm[2] +=  ['baseline' for _ in range(len(df['mean_osi_aneurysm'].tolist()[-5:-4]))]
                KER[2] +=  ['baseline' for _ in range(len(df['KER'].tolist()[:-4]))]
                VDR[2] +=  ['baseline' for _ in range(len(df['VDR'].tolist()[:-4]))]
                LSA[2] +=  ['baseline' for _ in range(len(df['LSA'].tolist()[:-4]))]
                HSA[2] +=  ['baseline' for _ in range(len(df['HSA'].tolist()[:-4]))]
                SCI[2] +=  ['baseline' for _ in range(len(df['SCI'].tolist()[:-4]))]
                ICI[2] +=  ['baseline' for _ in range(len(df['ICI'].tolist()[:-4]))]

    for case in cases:
        filenames = glob.glob(osp.join(res_dir, "csv", "hd", case, "*.csv"))
        for filename in filenames:
                df = pd.read_csv(filename, sep=' ')

                min_wss_aneurysm[0] += df['min_wss_aneurysm'].tolist()[:-4]
                max_wss_aneurysm[0] += df['max_wss_aneurysm'].tolist()[:-4]
                mean_wss_aneurysm[0] += df['mean_wss_aneurysm'].tolist()[:-4]
                min_osi_aneurysm[0] += df['min_osi_aneurysm'].tolist()[-5:-4]
                max_osi_aneurysm[0] += df['max_osi_aneurysm'].tolist()[-5:-4]
                mean_osi_aneurysm[0] += df['mean_osi_aneurysm'].tolist()[-5:-4]
                KER[0] += df['KER'].tolist()[:-4]
                VDR[0] += df['VDR'].tolist()[:-4]
                LSA[0] += df['LSA'].tolist()[:-4]
                HSA[0] += df['HSA'].tolist()[:-4]
                SCI[0] += df['SCI'].tolist()[:-4]
                ICI[0] += df['ICI'].tolist()[:-4]

                min_wss_aneurysm[1] +=  [case for _ in range(len(df['min_wss_aneurysm'].tolist()[:-4]))]
                max_wss_aneurysm[1] +=  [case for _ in range(len(df['max_wss_aneurysm'].tolist()[:-4]))]
                mean_wss_aneurysm[1] +=  [case for _ in range(len(df['mean_wss_aneurysm'].tolist()[:-4]))]
                min_osi_aneurysm[1] +=  [case for _ in range(len(df['min_osi_aneurysm'].tolist()[-5:-4]))]
                max_osi_aneurysm[1] +=  [case for _ in range(len(df['max_osi_aneurysm'].tolist()[-5:-4]))]
                mean_osi_aneurysm[1] +=  [case for _ in range(len(df['mean_osi_aneurysm'].tolist()[-5:-4]))]
                KER[1] +=  [case for _ in range(len(df['KER'].tolist()[:-4]))]
                VDR[1] +=  [case for _ in range(len(df['VDR'].tolist()[:-4]))]
                LSA[1] +=  [case for _ in range(len(df['LSA'].tolist()[:-4]))]
                HSA[1] +=  [case for _ in range(len(df['HSA'].tolist()[:-4]))]
                SCI[1] +=  [case for _ in range(len(df['SCI'].tolist()[:-4]))]
                ICI[1] +=  [case for _ in range(len(df['ICI'].tolist()[:-4]))]

                min_wss_aneurysm[2] +=  ['hd' for _ in range(len(df['min_wss_aneurysm'].tolist()[:-4]))]
                max_wss_aneurysm[2] +=  ['hd' for _ in range(len(df['max_wss_aneurysm'].tolist()[:-4]))]
                mean_wss_aneurysm[2] +=  ['hd' for _ in range(len(df['mean_wss_aneurysm'].tolist()[:-4]))]
                min_osi_aneurysm[2] +=  ['hd' for _ in range(len(df['min_osi_aneurysm'].tolist()[-5:-4]))]
                max_osi_aneurysm[2] +=  ['hd' for _ in range(len(df['max_osi_aneurysm'].tolist()[-5:-4]))]
                mean_osi_aneurysm[2] +=  ['hd' for _ in range(len(df['mean_osi_aneurysm'].tolist()[-5:-4]))]
                KER[2] +=  ['hd' for _ in range(len(df['KER'].tolist()[:-4]))]
                VDR[2] +=  ['hd' for _ in range(len(df['VDR'].tolist()[:-4]))]
                LSA[2] +=  ['hd' for _ in range(len(df['LSA'].tolist()[:-4]))]
                HSA[2] +=  ['hd' for _ in range(len(df['HSA'].tolist()[:-4]))]
                SCI[2] +=  ['hd' for _ in range(len(df['SCI'].tolist()[:-4]))]
                ICI[2] +=  ['hd' for _ in range(len(df['ICI'].tolist()[:-4]))]

    indicators = {
        'min_wss_aneurysm': min_wss_aneurysm,
        'max_wss_aneurysm': max_wss_aneurysm,
        'mean_wss_aneurysm': mean_wss_aneurysm,
        'min_osi_aneurysm': min_osi_aneurysm,
        'max_osi_aneurysm': max_osi_aneurysm,
        'mean_osi_aneurysm': mean_osi_aneurysm,
        'VDR': VDR,
        'LSA': LSA,
        'HSA': HSA,
        'SCI': SCI,
        'ICI': ICI,
    }

    titles = ['$WSS_{min}$', '$WSS_{max}$', '$WSS_{avg}$', '$OSI_{min}$', '$OSI_{max}$', '$OSI_{avg}$', 'KER', 'VDR', 'LSA', 'HSA', 'SCI', 'ICI']

    for title, key in zip(titles, indicators):
        data = {
            key: indicators[key][0],
            'Case': indicators[key][1],
            'Scenario': indicators[key][2],
        }
        df = pd.DataFrame(data=data)

        ax = sns.violinplot(data=df, x='Scenario', y=key, hue="Case", linewidth=1.5, palette="Set2")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel(title)
        fig = ax.get_figure()
        fig.savefig(osp.join(res_dir, "violin", f'{key}.png'))
        ax.clear()