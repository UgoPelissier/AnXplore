import os
import os.path as osp
import glob
import numpy as np
import pandas as pd
from alive_progress import alive_bar
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_dir = "data"
    cases = ['rigid', 'fsi']

    min_wss_aneurysm = [[], []]
    max_wss_aneurysm = [[], []]
    mean_wss_aneurysm = [[], []]
    mean_osi_aneurysm = [[], []]
    max_osi_aneurysm = [[], []]
    KER = [[], []]
    VDR = [[], []]
    LSA = [[], []]
    HSA = [[], []]
    SCI = [[], []]
    ICI = [[], []]

    for i, case in enumerate(cases):
        filenames = glob.glob(osp.join(data_dir, "csv", case, "*.csv"))
        with alive_bar(len(filenames), title=f"Extracting indicators for {case} case") as bar:
            for filename in filenames:
                df = pd.read_csv(filename, sep=' ')
                df.drop(df.columns[[0,1,4,5,6,7,9,10,11]], axis=1, inplace=True)
                min_wss_aneurysm[i] += df['min_wss_aneurysm'].tolist()[:-4]
                max_wss_aneurysm[i] += df['max_wss_aneurysm'].tolist()[:-4]
                mean_wss_aneurysm[i] += df['mean_wss_aneurysm'].tolist()[:-4]
                mean_osi_aneurysm[i] += df['mean_osi_aneurysm'].tolist()[-5:-4]
                max_osi_aneurysm[i] += df['max_osi_aneurysm'].tolist()[-5:-4]
                KER[i] += df['KER'].tolist()[:-4]
                VDR[i] += df['VDR'].tolist()[:-4]
                LSA[i] += df['LSA'].tolist()[:-4]
                HSA[i] += df['HSA'].tolist()[:-4]
                SCI[i] += df['SCI'].tolist()[:-4]
                ICI[i] += df['ICI'].tolist()[:-4]
                bar()
        print("Done.")
    
    min_wss_aneurysm = np.array(min_wss_aneurysm)
    max_wss_aneurysm = np.array(max_wss_aneurysm)
    mean_wss_aneurysm = np.array(mean_wss_aneurysm)
    mean_osi_aneurysm = np.array(mean_osi_aneurysm)
    max_osi_aneurysm = np.array(max_osi_aneurysm)
    KER = np.array(KER)
    VDR = np.array(VDR)
    LSA = np.array(LSA)
    HSA = np.array(HSA)
    SCI = np.array(SCI)
    ICI = np.array(ICI)

    min_wss_aneurysm = min_wss_aneurysm[1]/min_wss_aneurysm[0]
    max_wss_aneurysm = max_wss_aneurysm[1]/max_wss_aneurysm[0]
    mean_wss_aneurysm = mean_wss_aneurysm[1]/mean_wss_aneurysm[0]
    mean_osi_aneurysm = mean_osi_aneurysm[1]/mean_osi_aneurysm[0]
    max_osi_aneurysm = max_osi_aneurysm[1]/max_osi_aneurysm[0]
    KER = KER[1]/KER[0]
    VDR = VDR[1]/VDR[0]
    LSA = LSA[1]/LSA[0]
    HSA = HSA[1]/HSA[0]
    SCI = SCI[1]/SCI[0]
    ICI = ICI[1]/ICI[0]

    # Plot boxplots on same figure
    fig, ax = plt.subplots()
    bp = ax.boxplot([min_wss_aneurysm, max_wss_aneurysm, mean_wss_aneurysm], showfliers=False, patch_artist = True, notch ='True')

    color = '#5f6a6a'
 
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    
    # changing color of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linestyle =":")
    
    # changing colorof caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B')
    
    # changing color of medians
    for median in bp['medians']:
        median.set(color ='red')
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)

    ax.set_xticklabels(['min WSS', 'max WSS', 'mean WSS'])
    plt.title("FSI / Rigid - Aneurysm")
    plt.savefig(osp.join(data_dir, "wss.png"))

    # Plot boxplots on same figure
    fig, ax = plt.subplots()
    bp = ax.boxplot([mean_osi_aneurysm, max_osi_aneurysm], showfliers=False, patch_artist = True, notch ='True')

    color = '#5f6a6a'
 
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    
    # changing color of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linestyle =":")
    
    # changing colorof caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B')
    
    # changing color of medians
    for median in bp['medians']:
        median.set(color ='red')
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)

    ax.set_xticklabels(['mean OSI', 'max OSI'])
    plt.title("FSI / Rigid - Aneurysm")
    plt.savefig(osp.join(data_dir, "osi.png"))

    # Plot boxplots on same figure
    fig, ax = plt.subplots()
    bp = ax.boxplot([KER, VDR, LSA, HSA, SCI, ICI], showfliers=False, patch_artist = True, notch ='True')

    color = '#5f6a6a'
 
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    
    # changing color of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linestyle =":")
    
    # changing colorof caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B')
    
    # changing color of medians
    for median in bp['medians']:
        median.set(color ='red')
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)

    ax.set_xticklabels(['KER', 'VDR', 'LSA', 'HSA', 'SCI', 'ICI'])
    plt.title("FSI / Rigid")
    plt.savefig(osp.join(data_dir, "indicators.png"))