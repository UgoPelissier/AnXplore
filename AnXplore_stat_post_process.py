import os
import os.path as osp
import glob
import pandas as pd
import numpy as np
from alive_progress import alive_bar
import seaborn as sns
from scipy import stats

if __name__ == '__main__':

    res_dir = "res"
    ys = ['53', '54', '57', '60']
    cases = ['rigid', 'fsi']

    os.makedirs(osp.join(res_dir), exist_ok=True)
    os.makedirs(osp.join(res_dir, "violin"), exist_ok=True)
    os.makedirs(osp.join(res_dir, "violin", "baseline"), exist_ok=True)
    os.makedirs(osp.join(res_dir, "violin", "hd"), exist_ok=True)

    min_wss_aneurysm = [[] for _ in range(len(ys))]
    max_wss_aneurysm = [[] for _ in range(len(ys))]
    mean_wss_aneurysm = [[] for _ in range(len(ys))]
    min_osi_aneurysm = [[] for _ in range(len(ys))]
    max_osi_aneurysm = [[] for _ in range(len(ys))]
    mean_osi_aneurysm = [[] for _ in range(len(ys))]
    min_tawss_aneurysm = [[] for _ in range(len(ys))]
    max_tawss_aneurysm = [[] for _ in range(len(ys))]
    mean_tawss_aneurysm = [[] for _ in range(len(ys))]
    KER = [[] for _ in range(len(ys))]
    VDR = [[] for _ in range(len(ys))]
    LSA = [[] for _ in range(len(ys))]
    HSA = [[] for _ in range(len(ys))]
    SCI = [[] for _ in range(len(ys))]
    ICI = [[] for _ in range(len(ys))]

    for (i, y) in enumerate(ys):
        for case in cases:
            filenames = glob.glob(osp.join(res_dir, "csv", "baseline", y, case, "*.csv"))
            with alive_bar(len(filenames), title=f"Extracting indicators for {y} - {case} case") as bar:
                for filename in filenames:
                    df = pd.read_csv(filename, sep=' ')
                    min_wss_aneurysm[i] += df['min_wss_aneurysm'].tolist()[:-4]
                    max_wss_aneurysm[i] += df['max_wss_aneurysm'].tolist()[:-4]
                    mean_wss_aneurysm[i] += df['mean_wss_aneurysm'].tolist()[:-4]
                    min_osi_aneurysm[i] += df['min_osi_aneurysm'].tolist()[-5:-4]
                    max_osi_aneurysm[i] += df['max_osi_aneurysm'].tolist()[-5:-4]
                    mean_osi_aneurysm[i] += df['mean_osi_aneurysm'].tolist()[-5:-4]
                    min_tawss_aneurysm[i] += df['min_tawss_aneurysm'].tolist()[-5:-4]
                    max_tawss_aneurysm[i] += df['max_tawss_aneurysm'].tolist()[-5:-4]
                    mean_tawss_aneurysm[i] += df['mean_tawss_aneurysm'].tolist()[-5:-4]
                    KER[i] += df['KER'].tolist()[:-4]
                    VDR[i] += df['VDR'].tolist()[:-4]
                    LSA[i] += df['LSA'].tolist()[:-4]
                    HSA[i] += df['HSA'].tolist()[:-4]
                    SCI[i] += df['SCI'].tolist()[:-4]
                    ICI[i] += df['ICI'].tolist()[:-4]
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

    titles = ['$WSS_{min}$', '$WSS_{max}$', '$WSS_{mean}$', '$OSI_{min}$', '$OSI_{max}$', '$OSI_{mean}$', '$TAWSS_{min}$', '$TAWSS_{max}$', '$TAWSS_{mean}$', 'KER', 'VDR', 'LSA', 'HSA', 'SCI', 'ICI']

    means = [{} for _ in range(len(ys))]
    os.makedirs(osp.join(res_dir, "pvalues"), exist_ok=True)
    for (i, y) in enumerate(ys):
        with open(osp.join(res_dir, 'pvalues', f'pvalues_{y}.csv'), 'a') as f:
            f.write('Indicator\tRigid\tFSI\tP-value\n')
            for title, key in zip(titles, indicators):
                rigid = indicators[key][i][:int(len(indicators[key][i])/2)]
                fsi = indicators[key][i][int(len(indicators[key][i])/2):]

                rigid_mean = stats.describe(rigid).mean
                fsi_mean = stats.describe(fsi).mean
                pvalue = stats.ttest_ind(rigid, fsi).pvalue

                f.write(f'{key}\t{rigid_mean}\t{fsi_mean}\t{pvalue}\n')

                means[i][key] = [rigid_mean, fsi_mean, pvalue]
                
                # case = ['Rigid' for _ in range(int(len(indicators[key])/2))] + ['FSI' for _ in range(int(len(indicators[key])/2))]
                # data = {
                #     key: indicators[key],
                #     'Case': case
                # }
                # df = pd.DataFrame(data=data)

                # ax = sns.violinplot(data=df, y=key, hue="Case", linewidth=1.5, palette="Set2")
                # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                # ax.set_ylabel(title)
                # fig = ax.get_figure()
                # fig.savefig(osp.join(res_dir, "violin", "baseline", f'{key}.png'))
                # ax.clear()

    os.makedirs(osp.join(res_dir, "barplot"), exist_ok=True)
    for (i, y) in enumerate(ys):
        # Bar plot of means ratio
        data = {
            'Indicator': titles,
            'Ratio': [means[i][key][0]/means[i][key][1] for key in means[i]],
        }
        df = pd.DataFrame(data=data)
        df = df.set_index('Indicator')
        ax = df.plot.bar(rot=90)
        ax.set_ylabel('Ratio')
        # Bars in red
        for j in range(len(ax.patches)):
            ax.patches[j].set_color('r')
            # If p-value < 0.05, add a star
            if (means[i][list(means[i].keys())[j]][2] < 0.05):
                # ax.patches[j].set_color('g')
                ax.text(j, ax.patches[j].get_height() + 0.01, '*', fontsize=12, color='k')
        # No x label
        ax.set_xlabel('')
        # No legend
        ax.get_legend().remove()
        # Draw a horizontal line at 1
        ax.axhline(y=1, color='k', linestyle='-')
        # Horizontal grid in background
        ax.yaxis.grid(True)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(osp.join(res_dir, 'barplot', f'barplot_{y}.png'))
        ax.clear()
    
    exit(0)

    ########################################################################################################

    scenarios = ['baseline', 'hd']
    files = [osp.basename(file) for file in glob.glob(osp.join(res_dir, "csv", "hd", "rigid", "*.csv"))]

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

    for scenario in scenarios:
        for case in cases:
            filenames = [osp.join(res_dir, "csv", scenario, case, file) for file in files]
            with alive_bar(len(filenames), title=f"Extracting indicators for {scenario} - {case} case") as bar:
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

                        min_wss_aneurysm[2] +=  [scenario for _ in range(len(df['min_wss_aneurysm'].tolist()[:-4]))]
                        max_wss_aneurysm[2] +=  [scenario for _ in range(len(df['max_wss_aneurysm'].tolist()[:-4]))]
                        mean_wss_aneurysm[2] +=  [scenario for _ in range(len(df['mean_wss_aneurysm'].tolist()[:-4]))]
                        min_osi_aneurysm[2] +=  [scenario for _ in range(len(df['min_osi_aneurysm'].tolist()[-5:-4]))]
                        max_osi_aneurysm[2] +=  [scenario for _ in range(len(df['max_osi_aneurysm'].tolist()[-5:-4]))]
                        mean_osi_aneurysm[2] +=  [scenario for _ in range(len(df['mean_osi_aneurysm'].tolist()[-5:-4]))]
                        KER[2] +=  [scenario for _ in range(len(df['KER'].tolist()[:-4]))]
                        VDR[2] +=  [scenario for _ in range(len(df['VDR'].tolist()[:-4]))]
                        LSA[2] +=  [scenario for _ in range(len(df['LSA'].tolist()[:-4]))]
                        HSA[2] +=  [scenario for _ in range(len(df['HSA'].tolist()[:-4]))]
                        SCI[2] +=  [scenario for _ in range(len(df['SCI'].tolist()[:-4]))]
                        ICI[2] +=  [scenario for _ in range(len(df['ICI'].tolist()[:-4]))]
                        bar()
            print("Done.")

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
        fig.savefig(osp.join(res_dir, "violin", "hd", f'{key}.png'))
        ax.clear()