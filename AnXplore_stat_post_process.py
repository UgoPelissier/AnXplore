import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from scipy import stats

def extract_indicators(
        res_dir: str,
        yy: list[str],
        yyv: list[float],
        cases: list[str],
        ids: list[int]
) -> dict:
    mean_velocity_aneurysm = [[] for _ in range(len(yy))]

    mean_wss_aneurysm = [[] for _ in range(len(yy))]
    min_wss_aneurysm = [[] for _ in range(len(yy))]
    max_wss_aneurysm = [[] for _ in range(len(yy))]
    
    mean_osi_aneurysm = [[] for _ in range(len(yy))]
    max_osi_aneurysm = [[] for _ in range(len(yy))]
    
    mean_tawss_aneurysm = [[] for _ in range(len(yy))]
    min_tawss_aneurysm = [[] for _ in range(len(yy))]
    max_tawss_aneurysm = [[] for _ in range(len(yy))]
    
    KER = [[] for _ in range(len(yy))]
    VDR = [[] for _ in range(len(yy))]
    LSA = [[] for _ in range(len(yy))]
    HSA = [[] for _ in range(len(yy))]
    SCI = [[] for _ in range(len(yy))]
    ICI = [[] for _ in range(len(yy))]

    for (i, y) in enumerate(yy):
        for case in cases:
            with alive_bar(len(ids), title=f"Extracting indicators for {yyv[i]} - Case {case}") as bar:
                for id in ids:
                    df = pd.read_csv(os.path.join(res_dir, "csv", y, case, f"{id}.csv"), sep=' ')

                    mean_velocity_aneurysm[i] += df['mean_velocity_aneurysm'].tolist()

                    mean_wss_aneurysm[i] += df['mean_wss_aneurysm'].tolist()
                    min_wss_aneurysm[i] += df['min_wss_aneurysm'].tolist()
                    max_wss_aneurysm[i] += df['max_wss_aneurysm'].tolist()
                    
                    mean_osi_aneurysm[i] += df['mean_osi_aneurysm'].tolist()[-1:]
                    max_osi_aneurysm[i] += df['max_osi_aneurysm'].tolist()[-1:]

                    mean_tawss_aneurysm[i] += df['mean_tawss_aneurysm'].tolist()[-1:]    
                    min_tawss_aneurysm[i] += df['min_tawss_aneurysm'].tolist()[-1:]
                    max_tawss_aneurysm[i] += df['max_tawss_aneurysm'].tolist()[-1:]
                    
                    KER[i] += df['KER'].tolist()
                    VDR[i] += df['VDR'].tolist()
                    LSA[i] += df['LSA'].tolist()
                    HSA[i] += df['HSA'].tolist()
                    SCI[i] += df['SCI'].tolist()
                    ICI[i] += df['ICI'].tolist()
                    bar()
        print("Done.")

    indicators = {
        'mean_velocity_aneurysm': mean_velocity_aneurysm,
        'mean_wss_aneurysm': mean_wss_aneurysm,
        'min_wss_aneurysm': min_wss_aneurysm,
        'max_wss_aneurysm': max_wss_aneurysm,      
        'mean_osi_aneurysm': mean_osi_aneurysm,
        'max_osi_aneurysm': max_osi_aneurysm,
        'mean_tawss_aneurysm': mean_tawss_aneurysm,
        'min_tawss_aneurysm': min_tawss_aneurysm,
        'max_tawss_aneurysm': max_tawss_aneurysm,
        'KER': KER,
        'VDR': VDR,
        'LSA': LSA,
        # 'HSA': HSA,
        # 'SCI': SCI,
        'ICI': ICI,
    }

    return indicators

def compute_mean_ratio(
        res_dir: str,
        yy: list[str],
        yyv: list[float],
        indicators: dict,
        titles: list[str]
) -> dict:
    means = [{} for _ in range(len(yy))]
    os.makedirs(osp.join(res_dir, "pvalues"), exist_ok=True)
    os.makedirs(osp.join(res_dir, "plots"), exist_ok=True)
    for (i, y) in enumerate(yy):
        ratios = {}
        with open(osp.join(res_dir, 'pvalues', f'pvalues_{y}.csv'), 'a') as f:
            f.write('Indicator\tRatio\tP-value\n')
            for title, key in zip(titles, indicators):
                rigid = indicators[key][i][:int(len(indicators[key][i])/2)]
                fsi = indicators[key][i][int(len(indicators[key][i])/2):]
                ratio = []
                for (rig, fs) in zip(rigid, fsi):
                    if fs != 0:
                        ratio.append(rig/fs)
                    else:
                        ratio.append(0)
                ratios[key] = ratio

                rigid_mean = stats.describe(rigid).mean
                fsi_mean = stats.describe(fsi).mean
                ratio_mean = stats.describe(ratio).mean
                pvalue = stats.ttest_ind(rigid, fsi).pvalue

                f.write(f'{key}\t{ratio_mean}\t{pvalue}\n')

                means[i][key] = [ratio_mean, pvalue]

        plt.plot(ratios['mean_tawss_aneurysm'], ratios['mean_osi_aneurysm'], 'ro')
        # Draw line x = 1
        plt.plot([0.6, 1.4], [1, 1], 'k--')
        # Draw line y = 1
        plt.plot([1, 1], [0.6, 1.4], 'k--')
        # Limit axis
        plt.xlim(0.6, 1.4)
        plt.ylim(0.6, 1.4)
        plt.xlabel('TAWSS')
        plt.ylabel('OSI')
        plt.title(f'RIG/FSI - y={yyv[i]}')
        plt.tight_layout()
        plt.savefig(osp.join(res_dir, 'plots', f'tawss_osi_{y}.png'))

    return means

def barplot(
        res_dir: str,
        yy: list[str],
        means: dict,
        titles: list[str]
) -> None:
    os.makedirs(osp.join(res_dir, "barplot"), exist_ok=True)
    for (i, y) in enumerate(yy):
        # Bar plot of means ratio
        data = {
            'Indicator': titles,
            'Ratio': [means[i][key][0] for key in means[i]],
        }
        df = pd.DataFrame(data=data)
        df = df.set_index('Indicator')
        ax = df.plot.bar(rot=90)
        ax.set_ylabel('Ratio')
        # Bars in red
        for j in range(len(ax.patches)):
            ax.patches[j].set_color('r')
            # If p-value < 0.05, add a star
            if (means[i][list(means[i].keys())[j]][1] < 0.05):
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

if __name__ == '__main__':

    res_dir = "res"
    yy = ['80']
    yyv = [8.0]
    cases = ['rigid', 'fsi']

    start = 0
    end = 55
    exclude = [33, 85]
    ids = [i for i in range(start, end) if i not in exclude]

    titles = ['$\overline{V}$', '$\overline{WSS}$', '$WSS_{min}$', '$WSS_{max}$', '$\overline{OSI}$', '$OSI_{max}$', '$\overline{TAWSS}$', '$TAWSS_{min}$', '$TAWSS_{max}$', 'KER', 'VDR', 'LSA', 'ICI']

    indicators = extract_indicators(res_dir, yy, yyv, cases, ids)

    means = compute_mean_ratio(res_dir, yy, yyv, indicators, titles)

    barplot(res_dir, yy, means, titles)