import sys
import os.path as osp
import csv
from dataclasses import dataclass
import configparser
from glob import glob
import matplotlib.pyplot as plt
import os
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils, stuff


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


def read_config_params(base_dir, eval_dirs):
    gammas = []
    fine_tunned = []
    for eval in eval_dirs:
        eval_config = configparser.ConfigParser()
        eval_config.read(osp.join(osp.join(base_dir, eval), 'config.ini'))
        trained_model_dir = eval_config.get('DATA', 'TRAINED_MODEL_DIR')
        fine_tunned.append(eval_config.getboolean('DATA', 'FINE_TUNING'))

        train_config = configparser.ConfigParser()
        train_config.read(osp.join(trained_model_dir, 'config.ini'))
        gammas.append(train_config.getfloat('PARAMETERS', 'LOSS_PENALIZATION_GAMMA'))
    return gammas, fine_tunned


if __name__ == "__main__":
    base_dir = 'results'
    save_dir = path_utils.create_save_dir(base_dir, osp.join("plots", "test_acc"))
    logger = stuff.create_logger(save_dir)

    patients = [Patient('Child_73', es=14, ed=32 + 1), Patient('Adolescent_53', es=14, ed=28 + 1), Patient('Adult_11', es=15, ed=39 + 1)]

    # best_val_checkpoint
    eval_dirs = ['EVAL_20221212-233053',
                 'EVAL_20221212-233129',
                 'EVAL_20221213-194431']

    # checkpoint
    # eval_dirs = ['EVAL_20221213-195853',
    #              'EVAL_20221213-201241',
    #              'EVAL_20221213-201330']

    ft_dirs = [osp.sep.join(['TEST_FT', 'EVAL', x]) for x in sorted(os.listdir(osp.sep.join([base_dir, 'TEST_FT', 'EVAL'])))]
    eval_dirs.extend(ft_dirs)

    gammas, fine_tunned = read_config_params(base_dir, eval_dirs)

    fig, axs = plt.subplots(2, len(patients), constrained_layout=True, sharex='col', sharey='col', figsize=(8.5, 3.5), dpi=100)  # figsize=(5, 3)

    for i, patient in enumerate(patients):
        logger.info(patient.name)
        x = np.arange(abs(patient.es - patient.ed)).tolist()
        times = slice(patient.es, patient.ed, 1)
        print(x)
        print(times)
        labels = [None] * len(x)
        labels[0] = 'ES'
        labels[-1] = 'ED'
        plot_flow = False

        linewidth = 1.0
        markersize = 4
        for j, (eval, gamma, ft) in enumerate(zip(eval_dirs, gammas, fine_tunned)):
            eval_dir = osp.join(base_dir, eval)
            try:
                cnn_fwd_acc, cnn_bwd_acc, flow_fwd_acc, flow_bwd_acc = read_metric_file(osp.join(eval_dir, f'{patient.name}_acc.csv'))
                cnn_fwd_hd, cnn_bwd_hd, flow_fwd_hd, flow_bwd_hd = read_metric_file(osp.join(eval_dir, f'{patient.name}_hd.csv'))
            except FileNotFoundError:
                continue

            if not plot_flow:
                logger.info(
                    f'\t\tOF-ACC : [{(np.mean(np.array(flow_fwd_acc[times]).mean() + flow_bwd_acc[times]).mean())*0.5:.3f}, {np.array(flow_fwd_acc[times]).mean():.3f}, {np.array(flow_bwd_acc[times]).mean():.3f}]')
                logger.info(
                    f'\t\tOF-HD  : [{(np.mean(np.array(flow_fwd_hd[times]).mean() + flow_bwd_hd[times]).mean())*0.5:.3f}, {np.array(flow_fwd_hd[times]).mean():.3f}, {np.array(flow_bwd_hd[times]).mean():.3f}]')

                axs[0, i].set_title('Forward')
                axs[1, i].set_title('Backward')
                axs[0, i].plot(x, flow_fwd_acc[times], 'ro-', label='Forward OF', markersize=markersize, linewidth=linewidth)
                axs[1, i].plot(x, flow_bwd_acc[times], 'bo-', label='Backward OF', markersize=markersize, linewidth=linewidth)
                plot_flow = True

            if ft:
                lbl = 'FT '
                color = 'orange'
            else:
                lbl = 'CNN '
                color = 'green'

            if gamma == 0.1:
                linestyle = 'solid'
            elif gamma == 1.0:
                linestyle = 'dashed'
            elif gamma == 10.0:
                linestyle = 'dashdot'

            logger.info(f'\n[{j}]\t{eval}, Gamma: {gamma}')
            axs[0, i].plot(x, cnn_fwd_acc[times], color=color, linestyle=linestyle,
                           label=fr'{lbl}$\gamma=${gamma}', markersize=markersize, linewidth=linewidth)
            axs[1, i].plot(x, cnn_bwd_acc[times], color=color, linestyle=linestyle,
                           label=fr'{lbl}$\gamma=${gamma}', markersize=markersize, linewidth=linewidth)

            axs[0, i].set_xticks(x)
            axs[0, i].set_xticklabels(labels)
            axs[0, i].set_xlabel('Time')
            axs[0, i].set_ylabel('Forward')
            axs[1, i].set_xticks(x)
            axs[1, i].set_xticklabels(labels)
            axs[1, i].set_xlabel('Time')
            axs[1, i].set_ylabel('Backward')

            # box = axs[0].get_position()
            # axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # axs[0, i].legend(loc='center left')
            axs[0, i].grid(visible=True)

            # box = axs[1].get_position()
            # axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # axs[1, i].legend(loc='center left')
            axs[1, i].grid(visible=True)

            logger.info(
                f'\t\tCNN-ACC: [{(np.array(cnn_fwd_acc[times]).mean() + np.array(cnn_bwd_acc[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_acc[times]).mean():.3f}, {np.array(cnn_bwd_acc[times]).mean():.3f}]')
            logger.info(
                f'\t\tCNN-HD : [{(np.array(cnn_fwd_hd[times]).mean() + np.array(cnn_bwd_hd[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_hd[times]).mean():.3f}, {np.array(cnn_bwd_hd[times]).mean():.3f}]')
        logger.info('**************************************************')

    plt.savefig(osp.join(save_dir, 'test_acc.pdf'))

    # for patient in patients:
    #     logger.info(patient.name)
    #     # plt.figure(figsize=(6.5, 4.0), dpi=100)
    #     fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(3, 4), dpi=100)
    #     # ax = plt.subplot(111)
    #     x = np.arange(abs(patient.es + 1 - patient.ed)).tolist()
    #     times = slice(patient.es + 1, patient.ed)
    #     labels = [None] * len(x)
    #     labels[0] = 'ES'
    #     labels[-1] = 'ED'
    #     plot_flow = False

    #     for i, (eval, gamma) in enumerate(zip(eval_dirs, gammas)):
    #         eval_dir = osp.join(base_dir, eval)
    #         cnn_fwd_acc, cnn_bwd_acc, flow_fwd_acc, flow_bwd_acc = read_metric_file(osp.join(eval_dir, f'{patient.name}_acc.csv'))
    #         cnn_fwd_hd, cnn_bwd_hd, flow_fwd_hd, flow_bwd_hd = read_metric_file(osp.join(eval_dir, f'{patient.name}_hd.csv'))

    #         if not plot_flow:
    #             logger.info(
    #                 f'\t\tOF-ACC : [{(np.mean(np.array(flow_fwd_acc[times]).mean() + flow_bwd_acc[times]).mean())*0.5:.3f}, {np.array(flow_fwd_acc[times]).mean():.3f}, {np.array(flow_bwd_acc[times]).mean():.3f}]')
    #             logger.info(
    #                 f'\t\tOF-HD  : [{(np.mean(np.array(flow_fwd_hd[times]).mean() + flow_bwd_hd[times]).mean())*0.5:.3f}, {np.array(flow_fwd_hd[times]).mean():.3f}, {np.array(flow_bwd_hd[times]).mean():.3f}]')

    #             axs[0].set_title('Forward')
    #             axs[1].set_title('Backward')
    #             axs[0].plot(x, flow_fwd_acc[times], 'ro-', label='Forward OF', markersize=5)
    #             axs[1].plot(x, flow_bwd_acc[times], 'bo-', label='Backward OF', markersize=5)
    #             plot_flow = True

    #         logger.info(f'\n[{i}]\t{eval}, Gamma: {gamma}')
    #         axs[0].plot(x, cnn_fwd_acc[times], label=fr'$\gamma=${gamma}')
    #         axs[1].plot(x, cnn_bwd_acc[times], label=fr'$\gamma=${gamma}')

    #         logger.info(
    #             f'\t\tCNN-ACC: [{(np.array(cnn_fwd_acc[times]).mean() + np.array(cnn_bwd_acc[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_acc[times]).mean():.3f}, {np.array(cnn_bwd_acc[times]).mean():.3f}]')
    #         logger.info(
    #             f'\t\tCNN-HD : [{(np.array(cnn_fwd_hd[times]).mean() + np.array(cnn_bwd_hd[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_hd[times]).mean():.3f}, {np.array(cnn_bwd_hd[times]).mean():.3f}]')
    #     logger.info('**************************************************')

    #     axs[0].set_xticks(x)
    #     axs[0].set_xticklabels(labels)
    #     axs[0].set_xlabel('Time')
    #     axs[0].set_ylabel('Dice')
    #     axs[1].set_xticks(x)
    #     axs[1].set_xticklabels(labels)
    #     axs[1].set_xlabel('Time')
    #     axs[1].set_ylabel('Dice')

    #     # box = axs[0].get_position()
    #     # axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #     axs[0].legend(loc='center left')
    #     axs[0].grid(visible=True)

    #     # box = axs[1].get_position()
    #     # axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #     axs[1].legend(loc='center left')
    #     axs[1].grid(visible=True)

    #     plt.savefig(osp.join(save_dir, patient.name + '_acc.pdf'))

    # for patient in patients:
    #         logger.info(patient.name)
    #         # plt.figure(figsize=(6.5, 4.0), dpi=100)
    #         fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(3, 4), dpi=100)
    #         # ax = plt.subplot(111)
    #         x = np.arange(abs(patient.es + 1 - patient.ed)).tolist()
    #         times = slice(patient.es + 1, patient.ed)
    #         labels = [None] * len(x)
    #         labels[0] = 'ES'
    #         labels[-1] = 'ED'
    #         plot_flow = False

    #         for i, (eval, gamma) in enumerate(zip(eval_dirs, gammas)):
    #             eval_dir = osp.join(base_dir, eval)
    #             cnn_fwd_acc, cnn_bwd_acc, flow_fwd_acc, flow_bwd_acc = read_metric_file(osp.join(eval_dir, f'{patient.name}_acc.csv'))
    #             cnn_fwd_hd, cnn_bwd_hd, flow_fwd_hd, flow_bwd_hd = read_metric_file(osp.join(eval_dir, f'{patient.name}_hd.csv'))

    #             if not plot_flow:
    #                 logger.info(
    #                     f'\t\tOF-ACC : [{(np.mean(np.array(flow_fwd_acc[times]).mean() + flow_bwd_acc[times]).mean())*0.5:.3f}, {np.array(flow_fwd_acc[times]).mean():.3f}, {np.array(flow_bwd_acc[times]).mean():.3f}]')
    #                 logger.info(
    #                     f'\t\tOF-HD  : [{(np.mean(np.array(flow_fwd_hd[times]).mean() + flow_bwd_hd[times]).mean())*0.5:.3f}, {np.array(flow_fwd_hd[times]).mean():.3f}, {np.array(flow_bwd_hd[times]).mean():.3f}]')

    #                 axs[0].set_title('Forward')
    #                 axs[1].set_title('Backward')
    #                 axs[0].plot(x, flow_fwd_acc[times], 'ro-', label='Forward OF', markersize=5)
    #                 axs[1].plot(x, flow_bwd_acc[times], 'bo-', label='Backward OF', markersize=5)
    #                 plot_flow = True

    #             logger.info(f'\n[{i}]\t{eval}, Gamma: {gamma}')
    #             axs[0].plot(x, cnn_fwd_acc[times], label=fr'$\gamma=${gamma}')
    #             axs[1].plot(x, cnn_bwd_acc[times], label=fr'$\gamma=${gamma}')

    #             logger.info(
    #                 f'\t\tCNN-ACC: [{(np.array(cnn_fwd_acc[times]).mean() + np.array(cnn_bwd_acc[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_acc[times]).mean():.3f}, {np.array(cnn_bwd_acc[times]).mean():.3f}]')
    #             logger.info(
    #                 f'\t\tCNN-HD : [{(np.array(cnn_fwd_hd[times]).mean() + np.array(cnn_bwd_hd[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_hd[times]).mean():.3f}, {np.array(cnn_bwd_hd[times]).mean():.3f}]')
    #         logger.info('**************************************************')

    #         axs[0].set_xticks(x)
    #         axs[0].set_xticklabels(labels)
    #         axs[0].set_xlabel('Time')
    #         axs[0].set_ylabel('Dice')
    #         axs[1].set_xticks(x)
    #         axs[1].set_xticklabels(labels)
    #         axs[1].set_xlabel('Time')
    #         axs[1].set_ylabel('Dice')

    #         # box = axs[0].get_position()
    #         # axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #         axs[0].legend(loc='center left')
    #         axs[0].grid(visible=True)

    #         # box = axs[1].get_position()
    #         # axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #         axs[1].legend(loc='center left')
    #         axs[1].grid(visible=True)

    #         plt.savefig(osp.join(save_dir, patient.name + '_acc.pdf'))

    # for patient in patients:
    #         logger.info(patient.name)
    #         # plt.figure(figsize=(6.5, 4.0), dpi=100)
    #         fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(14, 9), dpi=100)
    #         # ax = plt.subplot(111)
    #         x = np.arange(abs(patient.es + 1 - patient.ed)).tolist()
    #         times = slice(patient.es + 1, patient.ed)
    #         labels = [None] * len(x)
    #         labels[0] = 'ES'
    #         labels[-1] = 'ED'
    #         plot_flow = False

    #         for i, (eval, gamma) in enumerate(zip(eval_dirs, gammas)):
    #             eval_dir = osp.join(base_dir, eval)
    #             cnn_fwd_acc, cnn_bwd_acc, flow_fwd_acc, flow_bwd_acc = read_metric_file(osp.join(eval_dir, f'{patient.name}_acc.csv'))
    #             cnn_fwd_hd, cnn_bwd_hd, flow_fwd_hd, flow_bwd_hd = read_metric_file(osp.join(eval_dir, f'{patient.name}_hd.csv'))

    #             if not plot_flow:
    #                 logger.info(
    #                     f'\t\tOF-ACC : [{(np.mean(np.array(flow_fwd_acc[times]).mean() + flow_bwd_acc[times]).mean())*0.5:.3f}, {np.array(flow_fwd_acc[times]).mean():.3f}, {np.array(flow_bwd_acc[times]).mean():.3f}]')
    #                 logger.info(
    #                     f'\t\tOF-HD  : [{(np.mean(np.array(flow_fwd_hd[times]).mean() + flow_bwd_hd[times]).mean())*0.5:.3f}, {np.array(flow_fwd_hd[times]).mean():.3f}, {np.array(flow_bwd_hd[times]).mean():.3f}]')

    #                 axs[0].set_title('Forward')
    #                 axs[1].set_title('Backward')
    #                 axs[0].plot(x, flow_fwd_acc[times], 'ro-', label='OF-FWD')
    #                 axs[1].plot(x, flow_bwd_acc[times], 'bo-', label='OF-BWD')
    #                 plot_flow = True

    #             logger.info(f'\n[{i}]\t{eval}, Gamma: {gamma}')
    #             axs[0].plot(x, cnn_fwd_acc[times], label=f'CNN-FWD-{gamma}')
    #             axs[1].plot(x, cnn_bwd_acc[times], label=f'CNN-BWD-{gamma}')

    #             logger.info(
    #                 f'\t\tCNN-ACC: [{(np.array(cnn_fwd_acc[times]).mean() + np.array(cnn_bwd_acc[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_acc[times]).mean():.3f}, {np.array(cnn_bwd_acc[times]).mean():.3f}]')
    #             logger.info(
    #                 f'\t\tCNN-HD : [{(np.array(cnn_fwd_hd[times]).mean() + np.array(cnn_bwd_hd[times]).mean())*0.5:.3f}, {np.array(cnn_fwd_hd[times]).mean():.3f}, {np.array(cnn_bwd_hd[times]).mean():.3f}]')
    #         logger.info('**************************************************')

    #         axs[0].set_xticks(x)
    #         axs[0].set_xticklabels(labels)
    #         axs[0].set_xlabel('Time')
    #         axs[0].set_ylabel('Dice')
    #         axs[1].set_xticks(x)
    #         axs[1].set_xticklabels(labels)
    #         axs[1].set_xlabel('Time')
    #         axs[1].set_ylabel('Dice')

    #         box = axs[0].get_position()
    #         axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #         axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #         axs[0].grid(visible=True)

    #         box = axs[1].get_position()
    #         axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #         axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #         axs[1].grid(visible=True)
    #         plt.savefig(osp.join(save_dir, patient.name + '_acc.pdf'))
