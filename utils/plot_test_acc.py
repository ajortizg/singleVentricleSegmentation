import sys
import os.path as osp
import csv
from dataclasses import dataclass
import configparser
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from utils import param_reader


@dataclass
class Patient:
    name: str
    es: int
    ed: int


def read_metric_file(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            row_f = [float(x) for x in row]
            yield row_f


def read_gammas(base_dir, eval_dirs):
    gammas = []
    for eval in eval_dirs:
        eval_config = configparser.ConfigParser()
        eval_config.read(osp.join(osp.join(base_dir, eval), 'config.ini'))

        train_config = configparser.ConfigParser()
        train_config.read(osp.join(param_reader.eval_params(eval_config)['trained_model_dir'], 'config.ini'))
        gammas.append(param_reader.train_params(train_config)['loss_gamma'])
    return gammas


if __name__ == "__main__":
    base_dir = 'results'
    save_dir = plots.createSaveDirectory(base_dir, 'PLOTS')
    logger = plots.create_logger(save_dir)

    patients = [Patient('Child_73', es=14, ed=32), Patient('Adolescent_53', es=14, ed=28), Patient('Adult_11', es=15, ed=39)]
    eval_dirs = ['EVAL_20221130-065110', 'EVAL_20221130-065202', 'EVAL_20221130-065314', 'EVAL_20221130-065346']
    gammas = read_gammas(base_dir, eval_dirs)

    for patient in patients:
        logger.info(patient.name)
        # plt.figure(figsize=(6.5, 4.0), dpi=100)
        fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(14, 9), dpi=100)
        # ax = plt.subplot(111)
        x = np.arange(abs(patient.es + 1 - patient.ed)).tolist()
        times = slice(patient.es + 1, patient.ed)
        labels = [None] * len(x)
        labels[0] = 'ES'
        labels[-1] = 'ED'
        plot_flow = False

        for i, (eval, gamma) in enumerate(zip(eval_dirs, gammas)):
            eval_dir = osp.join(base_dir, eval)
            cnn_fwd_acc, cnn_bwd_acc, flow_fwd_acc, flow_bwd_acc = read_metric_file(osp.join(eval_dir, f'{patient.name}_acc.csv'))
            cnn_fwd_hd, cnn_bwd_hd, flow_fwd_hd, flow_bwd_hd = read_metric_file(osp.join(eval_dir, f'{patient.name}_hd.csv'))

            if not plot_flow:
                logger.info(f'\t\tOF-ACC : [{(np.mean(np.array(flow_fwd_acc[times]).mean() + flow_bwd_acc[times]).mean())*0.5:.3f}, {np.array(flow_fwd_acc[times]).mean():.3f}, {np.array(flow_bwd_acc[times]).mean():.3f}]')
                logger.info(f'\t\tOF-HD  : [{(np.mean(np.array(flow_fwd_hd[times]).mean() + flow_bwd_hd[times]).mean())*0.5:.3f}, {np.array(flow_fwd_hd[times]).mean():.3f}, {np.array(flow_bwd_hd[times]).mean():.3f}]')

                axs[0].set_title('Forward')
                axs[1].set_title('Backward')
                axs[0].plot(x, flow_fwd_acc[times], 'ro-', label='OF-FWD')
                axs[1].plot(x, flow_bwd_acc[times], 'b*-', label='OF-BWD')
                plot_flow = True

            logger.info(f'\n[{i}]\t{eval}, Gamma: {gamma}')
            axs[0].plot(x, cnn_fwd_acc[times], label=f'CNN-FWD-{gamma}')
            axs[1].plot(x, cnn_bwd_acc[times], label=f'CNN-BWD-{gamma}')

            logger.info(f'\t\tCNN-ACC: [{(np.array(cnn_fwd_acc[times]).mean() + np.array(cnn_bwd_acc[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_acc[times]).mean():.3f}, {np.array(cnn_bwd_acc[times]).mean():.3f}]')
            logger.info(f'\t\tCNN-HD : [{(np.array(cnn_fwd_hd[times]).mean() + np.array(cnn_bwd_hd[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_hd[times]).mean():.3f}, {np.array(cnn_bwd_hd[times]).mean():.3f}]')
        logger.info('**************************************************')

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(labels)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Dice')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(labels)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Dice')

        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(osp.join(save_dir, patient.name + '_acc.pdf'))
