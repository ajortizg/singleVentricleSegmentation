import torch
import configparser
import sys
from dataset import SingleVentricleDataset, DatasetMode
from tqdm import tqdm
from warp import Warp
import os.path as osp
import torch.nn.functional as F
import csv
import torch.nn as nn
import transforms as T
from cnn_utils import BatchInfo
from loss import loss_func_complete

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from utils import torch_utils


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


data_transforms = T.ComposeUnary([T.Normalize()])
# data_transforms = T.ComposeUnary([T.Normalize(), T.PadTime(maxt=40)])
# data_mask_transforms = T.ComposeTernary([T.Resize(size=(14, 90, 90))])
# mask_transforms = T.ComposeUnary([T.Round(th=0.5)])
# flow_transforms = T.ComposeUnary([T.ResizeFlow3d(size=(14, 90, 90))])

ds = SingleVentricleDataset(config, DatasetMode.FULL, load_flow=True,
                            data_transforms=data_transforms,
                            mask_transforms=None,
                            data_mask_transforms=None,
                            flow_transforms=None)

save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'Warping')

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
pbar = tqdm(total=len(ds))
posp_masks = T.ComposeUnary([T.Normalize(), T.Erode()])
for (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in ds:
    NZ, NY, NX, NT = vol.shape
    grid = torch_utils.create_grid(NZ, NY, NX).unsqueeze(0).to(DEVICE)

    # patient_dir = plots.createSubDirectory(save_dir, pname)
    # plots.save_slices(m0, 'm0.png', patient_dir)
    # plots.save_slices(mk, 'mk.png', patient_dir)

    # warp = Warp(config, info)
    mts = [m0.to(DEVICE)]
    mtts = [mk.to(DEVICE)]
    ts = ff.shape[0]
    for t in range(ts):
        pbar.set_postfix_str(f'P: {pname}, S: {t}')

        # Forward mask propagation m0 -> mk
        # data_t = vol[:, :, :, init_ts + t].to(DEVICE)
        # plots.save_img_mask_slices(data_t,
        #                            posp_masks(mts[-1].cpu().detach()),
        #                            f'img_mask_t{init_ts + t}',
        #                            patient_dir,
        #                            color=[0, 0, 1],
        #                            alpha=0.5)
        # plots.save_compare_masks(
        #     data_t, plots.erode_mask(mts[-1]),
        #     plots.erode_mask(mtts[-1]),
        #     f'img_mask2_t_tt{init_ts + t}', patient_dir, color1=[0, 1, 0],
        #     color2=[0, 0, 1])
        u = ff[t, :, :, :, :].unsqueeze(0).to(DEVICE)
        # mt = warp(mts[-1], u)
        mt = torch_utils.warp(mts[-1].unsqueeze(0).unsqueeze(0), grid + u).squeeze()
        mts.append(mt)

        # Backward mask propagation mk -> m0
        # data_t = vol[:, :, :, final_ts - t].to(DEVICE)
        # plots.save_img_mask_slices(data_t,
        #                            posp_masks(mtts[-1].cpu().detach()),
        #                            f'img_mask_tt{init_ts + t}',
        #                            patient_dir,
        #                            color=[0, 1, 0],
        #                            alpha=0.5)
        u = bf[t, :, :, :, :].unsqueeze(0).to(DEVICE)
        # mtt = warp(mtts[-1], u)
        mtt = torch_utils.warp(mtts[-1].unsqueeze(0).unsqueeze(0), grid + u).squeeze()
        mtts.append(mtt)

    mtts.reverse()
    loss, l1, l2, l3 = loss_func_complete(mts, mtts)

    row = [pname]
    row.append('{:.2f}'.format(l1.item()))
    row.append('{:.2f}'.format(l2.item()))
    row.append('{:.2f}'.format(l3.item()))
    row.append('{:.2f}'.format(loss.item()))

    pbar.update(1)
    writer.writerow(row)

csv_file.close()
