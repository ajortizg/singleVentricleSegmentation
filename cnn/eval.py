import torch
import configparser
from tqdm import tqdm
from dataset import SingleVentricleDataset, DatasetMode
from warp import Warp
import torch.nn.functional as F
from cnn_utils import propagate
import sys
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from loss import loss_func_complete
import csv
import transforms

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from utils import torch_utils


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
USE_CUDA = config.get('DEVICE', 'CUDA_AVAILABLE')
if USE_CUDA and torch.cuda.is_available():
    DEVICE = 'cuda'
    CUDA_DEVICE = config.getint('DEVICE', 'cuda_device')
    torch.cuda.set_device(CUDA_DEVICE)
else:
    DEVICE = 'cpu'

# transf = [transforms.Normalize(mean=0.1478, std=0.1385)]
transf = [transforms.Normalize()]
ds = SingleVentricleDataset(config, DatasetMode.TRAIN, transf, load_flow=True)
save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN_EVAL')
# writer = SummaryWriter(log_dir=save_dir)

# save config file to save directory
conifg_output = osp.join(save_dir, 'config.ini')
with open(conifg_output, 'w') as config_file:
    config.write(config_file)

csv_file = open(osp.join(save_dir, 'cnn_diff.csv'), 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Patient', 'L1', 'L2', 'L3', 'LT', 'L1', 'L2', 'L3', 'LT'])

TRAINED_MODEL = config.get('DATA', 'TRAINED_MODEL')
net = torch.load(TRAINED_MODEL).to(DEVICE)
pbar = tqdm(total=len(ds))

net.eval()
with torch.no_grad():
    for (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in ds:
        NZ, NY, NX, NT = vol.shape
        nsize = (1, 1, NZ, NY, NX)

        warp = Warp(config, NZ, NY, NX)
        mts_cnn = [m0.reshape(nsize).to(DEVICE)]
        mtts_cnn = [mk.reshape(nsize).to(DEVICE)]

        mts_iw = [m0.to(DEVICE)]
        mtts_iw = [mk.to(DEVICE)]

        steps = len(ff)
        for t in range(steps):
            pbar.set_postfix_str(f'P: {pname}, S: {t+1}/{steps}')

            # Forward mask propagation m0 -> mk
            fwd_time = init_ts + t + 1
            data_t = vol[:, :, :, fwd_time].to(DEVICE)
            u = ff[t].to(DEVICE)
            mt_cnn = propagate(net, warp, data_t, nsize, u, mts_cnn[-1])
            mts_cnn.append(mt_cnn)
            mts_iw.append(warp(mts_iw[-1], u))

            # Backward mask propagation mk -> m0
            bwd_time = final_ts - t - 1
            data_t = vol[:, :, :, bwd_time].to(DEVICE)
            u = bf[t].to(DEVICE)
            mtt_cnn = propagate(net, warp, data_t, nsize, u, mtts_cnn[-1])
            mtts_cnn.append(mtt_cnn)
            mtts_iw.append(warp(mtts_iw[-1], u))

        mtts_cnn.reverse()
        mtts_iw.reverse()
        loss_cnn, l1_cnn, l2_cnn, l3_cnn = loss_func_complete(mts_cnn, mtts_cnn)
        loss_iw, l1_iw, l2_iw, l3_iw = loss_func_complete(mts_iw, mtts_iw)
        # writer.add_image(f'train_{pname}_mkt', torch.swapaxes(torch_utils.normalize(mt).squeeze(1), 0, 1), dataformats='NCHW')
        # writer.add_image(f'train_{pname}_m0tt', torch.swapaxes(torch_utils.normalize(mtt).squeeze(1), 0, 1), dataformats='NCHW')
        row = [pname]
        row.append('{:.2f}'.format(l1_cnn.item()))
        row.append('{:.2f}'.format(l2_cnn.item()))
        row.append('{:.2f}'.format(l3_cnn.item()))
        row.append('{:.2f}'.format(loss_cnn.item()))
        row.append('{:.2f}'.format(l1_iw.item()))
        row.append('{:.2f}'.format(l2_iw.item()))
        row.append('{:.2f}'.format(l3_iw.item()))
        row.append('{:.2f}'.format(loss_iw.item()))
        csv_writer.writerow(row)

        patient_dir = plots.createSubDirectory(save_dir, pname)
        fwd_dir = plots.createSubDirectory(patient_dir, 'fwd')
        bwd_dir = plots.createSubDirectory(patient_dir, 'bwd')

        for i in range(len(mts_cnn)):
            mt_cnn = torch_utils.normalize(mts_cnn[i].squeeze())
            mtt_cnn = torch_utils.normalize(mtts_cnn[i].squeeze())
            mt_iw = torch_utils.normalize(mts_iw[i])
            mtt_iw = torch_utils.normalize(mtts_iw[i])
            data_t = vol[:, :, :, init_ts + i].to(DEVICE)

            plots.save_compare_masks(
                data_t,
                plots.erode_mask(mt_cnn),
                plots.erode_mask(mt_iw),
                f'im_t_{init_ts + i}', fwd_dir,
                color1=[0, 1, 0],
                color2=[0, 0, 1], alpha=0.5)

            plots.save_compare_masks(
                data_t,
                plots.erode_mask(mtt_cnn),
                plots.erode_mask(mtt_iw),
                f'im_tt_{init_ts + i}', bwd_dir,
                color1=[0, 1, 0],
                color2=[0, 0, 1], alpha=0.5)

        pbar.update(1)

csv_file.close()
# writer.close()
