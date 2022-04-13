import torch
import configparser
import sys
from dataset import SingleVentricleDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet_3d import UNet3D
import os.path as osp

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
from torch_warping import create_grid, warp
import plots

# def plot_slices(X, str="", block=True):
#     """
#     X:  numpy array with shape [Z,Y,X]
#     """
#     fig, _ = plt.subplots(2, X.shape[0] // 2)
#     plt.suptitle(f"{str} {X.shape}")
#     for i, ax in enumerate(fig.get_axes()):
#         ax.imshow(X[i, :, :], cmap="gray", origin="lower")
#     plt.show(block=block)


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')
cuda_availabe = config.get('DEVICE', 'CUDA_AVAILABLE')
DEVICE = 'cuda:2' if cuda_availabe and torch.cuda.is_available() else 'cpu'
PATIENT_NAME = config.get('DATA', 'PATIENT_NAME')


ds = SingleVentricleDataset(config)
idx, found = ds.index_for_patient(PATIENT_NAME)
if not found:
    print(PATIENT_NAME + ' not found!')
    sys.exit()

(mask_systole, mask_diastole, systole_time, diastole_time, pname) = ds[idx]
fwd_of, bwd_of = ds.optflow_results_for_patient(PATIENT_NAME)
NZ, NY, NX = mask_systole.shape
print('================================================')
print('Load data for patient: ' + pname)
print(f'\t* (NZ, NY, NX) = ({NZ}, {NY}, {NX})')
print(f'\t* Systole at time: {systole_time}')
print(f'\t* Diastole at time: {diastole_time}')
print(f'\t* Optflows: {len(fwd_of)}, with shape: {fwd_of[0].shape}')
print('================================================')
init_timestep = min(diastole_time, systole_time)
final_timestep = max(diastole_time, systole_time)

# Mask initialization
# mask = None
# if init_timestep == systole_time:
#     mask = mask_systole.clone().to(DEVICE)
# else:
#     mask = mask_diastole.clone().to(DEVICE)

# mask.unsqueeze_(dim=0).unsqueeze_(dim=0)
# mask_systole.unsqueeze_(dim=0).unsqueeze_(dim=0).to(DEVICE)
m0 = mask_systole.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
mk = mask_diastole.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)


# Create grid of x,y,z coordinates for warping
grid = create_grid(NZ, NY, NX).unsqueeze(dim=0).to(DEVICE)


net = UNet3D(config).to(DEVICE)
net = torch.nn.DataParallel(net, device_ids=[2])
# loss_func = nn.MSELoss(reduction='sum')
# loss_func = nn.BCELoss(reduction='mean')
loss_func = nn.BCEWithLogitsLoss()
opt = optim.Adam(net.parameters(), lr=0.001)

mts = []
mtts = []
# create save directory
saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "CNN")
saveEpochDir = plots.createSubDirectory(saveDir, 'm0tt')
# plots.save_slices(m0.squeeze(), f"m0", saveEpochDir)
plots.save_single_zslices(m0.squeeze(), saveEpochDir, 'm0_gt', max_gray_value=1., color_channel=-1)


for e in range(1000):
    net.train()

    mts.append(m0)
    for j in range(len(fwd_of)):
        # Forward mask propagation
        u = fwd_of[j].unsqueeze(dim=0).to(DEVICE)
        mt = net(warp(mts[-1], grid + u))
        mt = F.sigmoid(mt)
        mts.append(mt)

    # mtts.append(mts[-1])
    mtts.append(mk)

    for j in range(len(bwd_of)):
        # Backward mask propagation
        u = bwd_of[j].unsqueeze(dim=0).to(DEVICE)
        mtt = net(warp(mtts[-1], grid + u))
        mtts.append(mtt)

    mtts.reverse()
    # plots.save_slices(mtts[0].squeeze(), f"m0tt_{e}", saveEpochDir)
    plots.save_single_zslices(F.sigmoid(mtts[0]).squeeze(), saveEpochDir,
                              f'm0_{e}', max_gray_value=1., color_channel=-1)

    total_loss = 0
    for k in range(len(mtts)):
        # mt = F.sigmoid(mts[k])

        loss = loss_func(mtts[k], mt)
        # loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
        total_loss += loss

    total_loss = total_loss / len(mtts)
    opt.zero_grad()
    total_loss.backward()
    opt.step()

    print(total_loss.item())
    mts.clear()
    mtts.clear()
