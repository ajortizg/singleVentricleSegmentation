import pandas as pd
import numpy as np
import os.path as osp
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils

if __name__ == '__main__':
    filesdir = 'results/eval/singleVentricleData_split/EVAL_20230517-050918'
    save_dir = path_utils.create_save_dir(osp.join('results', 'plots'), 'vol')
    print('Save dir: ', save_dir)
    patients = ['Child_73', 'Adolescent_53', 'Adult_11']
    pixdim = [[0.9875, 0.95, 0.8875], [0.8875, 0.925, 0.9875], [1.1125, 1.0125, 1.0375]]
    pixvol = [0.9875 * 0.95 * 0.8875, 0.8875 * 0.925 * 0.9875, 1.1125 * 1.0125 * 1.0375]

    metrics = ['vol', 'sur']
    ylabel = ["$cm^3$", "$cm^2$"]

    fig, axs = plt.subplots(2, len(patients), constrained_layout=True, sharex='col', sharey='none', figsize=(6, 2.0), dpi=100)  # figsize=(5, 3)
    linewidth = 1.0
    markersize = 1.0

    for i, patient in enumerate(patients):
        for j, metric in enumerate(metrics):
            df = pd.read_csv(osp.join(filesdir, f'{patient}_{metric}.csv'))
            gt = (df[f'{metric}gt'].to_numpy() * pixvol[i]) / 1000.0  # convert to cm3
            fwd = (df[f'{metric}fwd'].to_numpy() * pixvol[i]) / 1000.0
            bwd = (df[f'{metric}bwd'].to_numpy() * pixvol[i]) / 1000.0

            assert gt.size == fwd.size and gt.size == bwd.size

            # plt.style.use('seaborn-paper')
            labels = [None] * len(gt)
            labels[0] = 'ES'
            labels[-1] = 'ED'

            axs[j, i].plot(gt, 'r', label='Ground-truth', linewidth=linewidth, markersize=markersize)
            axs[j, i].plot(fwd, 'g', label='Forward', linewidth=linewidth, markersize=markersize)
            axs[j, i].plot(bwd, 'b', label='Backward', linewidth=linewidth, markersize=markersize)
            axs[j, i].set_xticks(np.arange(gt.size))
            axs[j, i].set_xticklabels(labels)
            # if j == 0:
            #     axs[j,i].legend()
            # if j == 0:
            #     axs[j, i].set_title(patient.split('_')[0])
            axs[j, i].set_xlabel('Time')
            axs[j, i].set_ylabel(ylabel[j])
            axs[j, i].grid(visible=True, linewidth=0.1)

    plt.savefig(osp.join(save_dir, 'vol.pdf'))
