import torch
import configparser
from tqdm import tqdm
from dataset import SingleVentricleDataset
from warp import Warp
import sys
import os.path as osp
import csv

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
import plots
import torch_utils


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
USE_CUDA = config.get('DEVICE', 'CUDA_AVAILABLE')
if USE_CUDA and torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_device(2)
else:
    DEVICE = 'cpu'

train_ds = SingleVentricleDataset(config, load_flow=True)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN_EVAL')

TRAINED_MODEL = config.get('DATA', 'TRAINED_MODEL')
SHUFFLE = config.getboolean('PARAMETERS', 'SHUFFLE')

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

csv_file = open(osp.join(save_dir, 'cnn_diff.csv'), 'w')
writer = csv.writer(csv_file)
mean_diff = 0

net = torch.load(TRAINED_MODEL).to(DEVICE)
pbar = tqdm(total=len(train_ds))
idxs = torch.randperm(len(train_ds)) if SHUFFLE else torch.arange(len(train_ds))
for idx in idxs:
    net.eval()
    with torch.no_grad():
        (pname, vol, mask_syst, mask_diast, tsyst, tdias, ff, bf) = train_ds[idx]
        pbar.set_postfix_str(f'P: {pname}')
        vol = torch_utils.normalize(vol)
        NZ, NY, NX, NT = vol.shape

        init_ts = min(tdias, tsyst)
        final_ts = max(tdias, tsyst)
        steps = abs(init_ts - final_ts)

        # Mask initialization
        m0, mk = None, None
        if init_ts == tsyst:
            m0 = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
        else:
            m0 = mask_diast.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            mk = mask_syst.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)

        patient_dir = plots.createSubDirectory(save_dir, pname)
        plots.save_slices(m0.squeeze(), 'm0.png', patient_dir)
        plots.save_slices(mk.squeeze(), 'mk.png', patient_dir)

        warp = Warp(config, NZ, NY, NX)
        mts = [m0]
        mtts = [mk]
        bf.reverse()
        for t in range(steps):
            pbar.set_postfix_str(f'P: {pname}, D: {mean_diff:.2f}, S: {t+1}/{steps}')

            # Forward mask propagation m0 -> mk
            fwd_time = init_ts + t + 1
            data_t = vol[:, :, :, fwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = ff[t].to(DEVICE)
            # mt = warp(mt.squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            mt = warp(mts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            x = torch.cat((data_t, mt), dim=1)
            mt = net(x)
            plots.save_slices(mt.squeeze(), f'mt{fwd_time}.png', patient_dir)
            mts.append(mt)

            # Backward mask propagation mk -> m0
            bwd_time = final_ts - t - 1
            data_t = vol[:, :, :, bwd_time].unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
            u = bf[t].to(DEVICE)
            # mtt = warp(mtt.squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            mtt = warp(mtts[-1].squeeze(), u).unsqueeze(dim=0).unsqueeze(dim=0)
            x = torch.cat((data_t, mtt), dim=1)
            mtt = net(x)
            plots.save_slices(mtt.squeeze(), f'mtt{bwd_time}.png', patient_dir)
            mtts.append(mtt)

        mtts.reverse()
        row = [pname]
        total_diff = 0.0
        for k in range(len(mtts)):
            diff = torch.abs(mts[k] - mtts[k]).squeeze() / number_of_voxels
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
