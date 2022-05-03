import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from tqdm import tqdm
from warp import Warp
import time
import os.path as osp
import matplotlib.pyplot as plt
import torch.functional as F
from torch import linalg as la
import csv

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
import plots
import torch_utils


def normalize(x):
    # Normalize between 0 and 1
    min = torch.min(x)
    max = torch.max(x)
    return (x - min) / (max - min)


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
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping')

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')
idx, found = train_ds.index_for_patient(PATIENT_NAME)
if not found:
    print(PATIENT_NAME + " not found!")
    sys.exit()

csv_file = open(osp.join(save_dir, 'diff.csv'), 'w')
writer = csv.writer(csv_file)
mean_diff = 0
pbar = tqdm(total=len(train_ds))
# for (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) in train_ds:
(pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) = train_ds[idx]
NZ, NY, NX, NT = vol.shape
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
# steps = abs(init_ts - final_ts)

# Mask initialization
m0, mk = None, None
if init_ts == tsyst:
    m0 = mask_syst
    mk = mask_diast
    # print('m0 = mask_systole', '\tmk = mask_diastole')
else:
    m0 = mask_diast
    mk = mask_syst
    # print('m0 = mask_diastole', '\tmk = mask_systole')

patient_dir = plots.createSubDirectory(save_dir, pname)
plots.save_slices(m0, 'm0.png', patient_dir)
plots.save_slices(mk, 'mk.png', patient_dir)

warp = Warp(config, NZ, NY, NX)
mts = [m0.to(DEVICE)]
mtts = [mk.to(DEVICE)]
bf.reverse()
# for t in range(steps):
for t in range(len(ff)):
    pbar.set_postfix_str(f'P: {pname}, D: {mean_diff:.2f}, S: {t+1}/{len(ff)}')
    # Forward mask propagation m0 -> mk
    u = ff[t].to(DEVICE)

    # NZ, NY, NX, _ = u.shape
    # for z in range(NZ):
    #     for y in range(NY):
    #         for x in range(NX):
    #             uvw = u[z, y, x, :]
    #             n_uvw = la.norm(uvw)
    #             # print(n_uvw)
    #             if n_uvw > 5.0:
    #                 u[z, y, x, :] = torch.zeros_like(uvw)
    th = 0.6
    # nm = la.norm(u, dim=-1)
    # u[nm > th, :] = torch.zeros(3, device=DEVICE)
    mt = warp(mts[-1], u)
    # mt = torch_utils.warp(mts[-1].unsqueeze(dim=0).unsqueeze(dim=0), (grid + u).unsqueeze(dim=0), mode='bilinear').squeeze()
    # mt = normalize(mt)
    mt = torch.where(mt > 0.5, 1.0, 0.0)
    plots.save_slices(mt, f'mt{init_ts+t+1}.png', patient_dir)
    # plots.save_slices(mt, f'mt{t+1}.png', patient_dir)
    mts.append(mt)

    # Backward mask propagation mk -> m0
    u = bf[t].to(DEVICE)
    # nm = la.norm(u, dim=-1)
    # u[nm > th, :] = torch.zeros(3, device=DEVICE)

    mtt = warp(mtts[-1], u)
    # mtt = torch_utils.warp(mtts[-1].unsqueeze(dim=0).unsqueeze(dim=0), (grid + u).unsqueeze(dim=0), mode='bilinear').squeeze()
    # mtt = normalize(mtt)
    mtt = torch.where(mtt > 0.5, 1.0, 0.0)
    plots.save_slices(mtt, f'mtt{final_ts-t-1}.png', patient_dir)
    # plots.save_slices(mtt, f'mtt{t-1}.png', patient_dir)
    mtts.append(mtt)

mtts.reverse()
row = [pname]
total_diff = 0.0
for k in range(len(mtts)):
    diff = torch.abs(mts[k] - mtts[k])
    ndiff = diff.norm().item()
    total_diff += ndiff
    # print(f"|diff| = {ndiff}")
    row.append('{:.2f}'.format(ndiff))
    plots.save_slices(diff, f'Diff_mt_mtt_t{k}.png', patient_dir)
    plots.save_single_zslices(diff, patient_dir, f'Diff_{k}', 1., 0)

pbar.update(1)
writer.writerow(row)
mean_diff = total_diff / len(mtts)
# print(f'mean |diff| =  {mean_diff}')

csv_file.close()
