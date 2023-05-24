import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import sys
import matplotlib

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils

if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 8})

    save_dir = path_utils.create_save_dir("results/plots", "lambda")
    print("Save dir: ", save_dir)

    lambdas = np.array([0.005, 0.05, 0.5, 5, 50, 500, 5000])
    accs = np.array([0.89 , 0.892, 0.895, 0.897, 0.894, 0.888, 0.861])

    plt.figure(figsize=(3, 2), dpi=100, constrained_layout=True)
    plt.plot(lambdas, accs, label='cnn')
    plt.xlabel('$\lambda$')
    plt.xscale("log")
    plt.ylabel('Dice')
    plt.grid(visible=True, linewidth=0.1)
    plt.axvline(x=5, linestyle='dashed', color="red", linewidth=0.6)
    plt.savefig(osp.join(save_dir, 'lambdas_exp.pdf'))
