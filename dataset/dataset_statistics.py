import configparser
from tqdm import tqdm
import sys
import torch
import os.path as osp
import numpy as np
from singleVentricleDataset import SingleVentricleDataset

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.transforms as T

config = configparser.ConfigParser()
config.read('parser/configPreprocessing.ini')

# transf = T.ComposeUnary([T.Standarize(mean=4131.408831052098, std=7339.299260790935)])
# transf = T.ComposeUnary([T.Normalize()])
# transf = T.ComposeUnary([T.Normalize(min=-4972.10205078125, max=266556.09375)])

ds_train = SingleVentricleDataset(config, mode='train')
ds_val = SingleVentricleDataset(config, mode='val')

N = len(ds_train) + len(ds_val)
pbar = tqdm(total=N)
sum = 0
squared_sum = 0
K = 0
maximum = 0
minimum = 1e10
for ds in [ds_train, ds_val]:
    for idx in range(len(ds)):
        patient = ds[idx]
        pbar.set_postfix_str(f'P: {patient.name}')

        # data = transf(patient.nii_data_xyzt)
        data = patient.nii_data_xyzt

        max = np.max(data)
        if max > maximum:
            maximum = max

        min = np.min(data)
        if min < minimum:
            minimum = min

        K += abs(patient.tDiastole - patient.tSystole)
        sum += np.mean(data)
        squared_sum += np.mean(data**2)
        pbar.update(1)


mean = sum / N
std = (squared_sum / N - mean**2)**0.5
l = (K / N) - 1
print(f'mean: {mean}\nstd: {std}\nlambda: {l}\nmin: {minimum}\nmax: {maximum}')
