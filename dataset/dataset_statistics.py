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

transf = T.ComposeUnary([T.Normalize(mean=0.04717000863622656, std=0.08217189410007013)])
# transf = T.ComposeUnary([T.Normalize()])

ds = SingleVentricleDataset(config)
pbar = tqdm(total=len(ds))
sum = 0
squared_sum = 0
K = 0
for idx in range(len(ds)):
    patient = ds[idx]
    pbar.set_postfix_str(f'P: {patient.name}')

    data = transf(torch.from_numpy(patient.nii_data_zyxt)).numpy()

    K += abs(patient.tDiastole - patient.tSystole)
    sum += np.mean(data)
    squared_sum += np.mean(data**2)
    pbar.update(1)

mean = sum / len(ds)
std = (squared_sum / len(ds) - mean**2)**0.5
l = (K / len(ds)) - 1
print(f'mean: {mean}, std: {std}, lambda: {l}')
