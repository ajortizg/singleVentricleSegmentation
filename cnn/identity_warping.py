import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from tqdm import tqdm
from warp import Warp
import time
import os.path as osp
import matplotlib.pyplot as plt

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
# from torch_warping import create_grid, warp
import plots


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
DEVICE = 'cuda' if cuda_availabe and torch.cuda.is_available() else 'cpu'
PIN_MEMORY = False if DEVICE == 'cpu' else True
BATCH_SIZE = config.getint('PARAMETERS', 'BATCH_SIZE')
NUM_EPOCHS = config.getint('PARAMETERS', 'NUM_EPOCHS')
VERBOSE = config.getboolean('DEBUG', 'VERBOSE')

train_ds = SingleVentricleDataset(config, load_flow=True)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'WARPING')

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

iter = 0
tic = time.time()
for e in tqdm(range(NUM_EPOCHS)):
    # Loop over the training set
    for (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) in train_ds:
        NZ, NY, NX, NT = vol.shape

        if VERBOSE:
            print("\n")
            print("===========================================================")
            print(f'Load data for patient: {pname}')
            print(f'\t* (NZ, NY, NX, NT) = ({NZ}, {NY}, {NX}, {NT})')
            print(f'\t* masks: {mask_syst.shape}, { mask_diast.shape}')
            print(f'\t* Systole at time: {tsyst}')
            print(f'\t* Diastole at time: {tdias}')
            print(f'\t* Optflows: {len(ff)}, with shape: {ff[0].shape}')
            print("===========================================================")

        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)

        # Mask initialization
        m0, mk = None, None
        if init_ts == tsyst:
            m0 = mask_syst.clone().to(DEVICE)
            mk = mask_diast.clone().to(DEVICE)
            print('m0 = mask_systole', '\tmk = mask_diastole')
        else:
            m0 = mask_diast.clone().to(DEVICE)
            mk = mask_syst.clone().to(DEVICE)
            print('m0 = mask_diastole', '\tmk = mask_systole')

        patient_dir = plots.createSubDirectory(save_dir, pname)
        # plots.save_single_zslices(m0.cpu().detach().squeeze(), patient_dir, 'm0')
        # plots.save_single_zslices(mk.cpu().detach().squeeze(), patient_dir, 'mk')
        plots.save_slices(m0.cpu().detach(), 'm0.png', patient_dir)
        plots.save_slices(mk.cpu().detach(), 'mk.png', patient_dir)

        warp = Warp(config, NZ, NY, NX)

        mts = [m0]
        k = 0
        for t in range(init_ts, final_ts, 1):
            u = ff[k].to(DEVICE)
            mt = warp(mts[-1], u)
            # plots.save_single_zslices(mt.cpu().detach().squeeze(), patient_dir, f'mt{t}')
            plots.save_slices(mt, f'mt{t+1}.png', patient_dir)
            mts.append(mt)
            k += 1

        mtts = [mk]
        bf.reverse()
        k = 0
        for t in range(final_ts, init_ts, -1):
            u = bf[k].to(DEVICE)
            mtt = warp(mtts[-1], u)
            plots.save_slices(mtt, f'mtt{t-1}.png', patient_dir)
            mtts.append(mtt)
            k += 1

        mtts.reverse()
        total_diff = 0.0
        for k in range(len(mtts)):
            diff = torch.abs(mts[k] - mtts[k])
            ndiff = diff.norm().item()
            total_diff += ndiff
            print(f"|diff| = {ndiff}")

            plots.save_slices(diff, f"Diff_mt_mtt_t{k}.png", patient_dir)
            plots.save_single_zslices(diff, patient_dir, "Diff", 1., 0)
        print(f'mean diff =  {total_diff / len(mtts)}')
