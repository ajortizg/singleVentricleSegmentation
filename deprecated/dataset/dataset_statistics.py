import configparser
from tqdm import tqdm
import sys
import torch
import os.path as osp
import numpy as np
from singleVentricleDataset import SingleVentricleDataset
import matplotlib.pyplot as plt
import yaml
from scipy import ndimage

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
# import utils.quaternary_transforms as T
from utilities import plots

config = configparser.ConfigParser()
config.read('parser/configPreprocessing.ini')

# save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Statistics')
# hists_dir = plots.createSubDirectory(save_dir, 'histograms')
# save config file to save directory
# conifg_output = osp.join(save_dir, 'config.ini')
# with open(conifg_output, 'w') as config_file:
#     config.write(config_file)

# transf = T.ComposeUnary([T.Standarize(mean=4167.181616023748, std=7324.106522252932)])
# transf = T.ComposeUnary([T.Normalize()])
# transf = T.ComposeUnary([T.Normalize(min=-6418.61376953125, max=269167.3125)])

ds = SingleVentricleDataset(config, mode='full')
# ds_val = SingleVentricleDataset(config, mode='val')

# N = len(ds_train) + len(ds_val)
pbar = tqdm(total=len(ds))
# sum = 0
# squared_sum = 0
# K = 0
# max_global = 0
# min_global = 1e10

# max_list = []
# min_list = []
# mean_list = []
# std_list = []

# for ds in [ds_train, ds_val]:
shapes = np.zeros((len(ds), 4))

for idx in range(len(ds)):
    patient = ds[idx]
    pbar.set_postfix_str(f'P: {patient.name}')

    # print(patient.nii_data_zyxt.shape)
   
    shapes[idx] = np.array(patient.nii_data_zyxt.shape)

    # init_ts = min(patient.tDiastole, patient.tSystole)
    # final_ts = max(patient.tDiastole, patient.tSystole)

    # data = patient.nii_data_zyxt[:, :, :, init_ts:final_ts + 1]
    # # data = patient.nii_data_xyzt

    # # max_local = np.max(data)
    # # if max_local > max_global:
    # #     max_global = max_local

    # # min_local = np.min(data)
    # # if min_local < min_global:
    # #     min_global = min_local

    # mean = np.mean(data)
    # std = np.std(data)

    # # max_list.append(max_local)
    # # min_list.append(min_local)
    # mean_list.append(mean)
    # std_list.append(std)

    # # hist = ndimage.histogram(data, min_local, max_local, 100)

    # # plt.bar(np.arange(len(hist)), hist)
    # # plt.title(f'min: {float(min_local):,.2f}, max: {float(max_local):,.2f}, \nmean: {float(mean):,.2f}, std: {float(std):,.2f}')
    # # plt.savefig(osp.join(hists_dir, patient.name + '.png'), dpi=100)
    # # plt.close('all')

    # K += abs(patient.tDiastole - patient.tSystole)
    # sum += np.mean(data)
    # squared_sum += np.mean(data**2)
    pbar.update(1)

i = 3
print(np.min(shapes[:,i]))
print(np.max(shapes[:,i]))

# mean = sum / N
# std = (squared_sum / N - mean**2)**0.5
# l = (K / N) - 1
# # print(f'mean: {mean}\nstd: {std}\nlambda: {l}\nmin: {min_global}\nmax: {max_global}')
# print(f'mean: {mean}\nstd: {std}\nlambda: {l}')

# x = np.arange(N)
# mean_array = np.asarray(mean_list)
# std_array = np.asarray(std_list)
# # min_array = np.asarray(min_list)
# # max_array = np.asarray(max_list)

# # plt.errorbar(x, mean_array, std_array, fmt='ok', lw=3)
# # plt.errorbar(x, mean_array, [mean_array - min_array, max_array - mean_array],
# #              fmt='.k', ecolor='gray', lw=1)
# # plt.xlim(-1, N)
# # plt.savefig(osp.join(save_dir, 'data_dist.png'), dpi=100)
# # plt.close('all')

# data = {}
# data['mean'] = float(mean)
# data['std'] = float(std)
# data['lambda'] = float(l)
# # data['min'] = float(min_global)
# # data['max'] = float(max_global)
# with open(osp.join(save_dir, 'stats.yaml'), 'w') as f:
#     yaml.dump(data, f)
