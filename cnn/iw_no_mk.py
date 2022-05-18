import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from tqdm import tqdm
from warp import Warp
import time
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch.functional as F
from torch import linalg as la
import csv

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
import plots
import torch_utils


print("\n\n")
print("===========================================================")
print("===========================================================")
print("                   Identity warping:")
print("===========================================================")
print("===========================================================")
print("\n\n")


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
if cuda_availabe and torch.cuda.is_available():
    DEVICE = 'cuda'
    CUDA_DEVICE = config.getint('DEVICE', 'CUDA_DEVICE')
    torch.cuda.set_device(CUDA_DEVICE)
else:
    DEVICE = 'cpu'


train_ds = SingleVentricleDataset(config, load_flow=True)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping_mi_to_m0')

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

# PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
# idx, found = train_ds.index_for_patient(PATIENT_NAME)
# if not found:
#     print(PATIENT_NAME + " not found!")
#     sys.exit()

csv_file = open(osp.join(save_dir, 'diff.csv'), 'w')
writer = csv.writer(csv_file)
mean_diff = 0
pbar = tqdm(total=len(train_ds))
for (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) in train_ds:
    # if pname != PATIENT_NAME:
    #     continue
    # (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) = train_ds[idx]
    NZ, NY, NX, NT = vol.shape
    vol = torch_utils.normalize(vol)
    # grid = torch_utils.create_grid(NZ, NY, NX).to(DEVICE)

    # print("\n")
    # print("===========================================================")
    # print(f'Load data for patient: {pname}')
    # print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
    # print(f'\t* masks: {mask_syst.shape}, { mask_diast.shape}')
    # print(f'\t* Systole at time: {tsyst}')
    # print(f'\t* Diastole at time: {tdias}')
    # print(f'\t* Optflows: {len(ff)}, with shape: {ff[0].shape}')
    # print("===========================================================")

    init_ts = min(tdias, tsyst)
    final_ts = max(tdias, tsyst)

    # Mask initialization
    m0, mk = None, None
    if init_ts == tsyst:
        m0 = mask_syst
        mk = mask_diast
    else:
        m0 = mask_diast
        mk = mask_syst

    patient_dir = plots.createSubDirectory(save_dir, pname)
    plots.save_slices(m0, 'm0.png', patient_dir)
    plots.save_slices(mk, 'mk.png', patient_dir)

    warp = Warp(config, NZ, NY, NX)
    mts = [m0.to(DEVICE)]

    # Forward mask propagation m0 -> mk
    for t in range(len(ff)):
        pbar.set_postfix_str(f'P: {pname}, D: {mean_diff:.2f}, S: {t+1}/{len(ff)}')

        data_t = vol[:, :, :, init_ts + t].to(DEVICE)
        plots.save_img_mask_slices(data_t, mts[-1], f'img_mask_{init_ts + t}', patient_dir)
        u = ff[t].to(DEVICE)
        mt = warp(mts[-1], u)
        plots.save_slices(mt, f'mt{init_ts + t + 1}.png', patient_dir)
        mts.append(mt)  # m0, m1, m2, ..., m9

    row = [pname]
    # Backward mask propagation mi -> m0
    for i in range(1, len(mts)):
        mi = mts[i]
        flow_idxs = np.arange(i - 1, -1, -1)
        for f in flow_idxs:
            u = bf[f].to(DEVICE)
            mi = warp(mi, u)
        plots.save_slices(mi, f'mt0_{i}.png', patient_dir)
        data_t = vol[:, :, :, init_ts].to(DEVICE)
        plots.save_img_mask_slices(data_t, mi, f'img_mask_0_{i}', patient_dir)

        diff = torch.abs(mts[0] - mi)
        ndiff = diff.norm().item()
        row.append('{:.2f}'.format(ndiff))
        plots.save_colorbar_slices(diff, f'Diff_m0_mt0_{i}.png', patient_dir)
    
    writer.writerow(row)
    pbar.update(1)

csv_file.close()
