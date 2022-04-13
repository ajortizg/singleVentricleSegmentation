import torch
import configparser
import sys
from dataset import SingleVentricleDataset
import torch.optim as optim
from unet_3d import UNet3D
import os.path as osp

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../utils')))
from torch_warping import create_grid, warp
import plots

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
m0, mk = None, None
if init_timestep == systole_time:
    m0 = mask_systole.clone().unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
    mk = mask_diastole.clone().unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
else:
    m0 = mask_diastole.clone().unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
    mk = mask_systole.clone().unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)

# mask.unsqueeze_(dim=0).unsqueeze_(dim=0)
# mask_systole.unsqueeze_(dim=0).unsqueeze_(dim=0).to(DEVICE)
# m0 = mask_systole.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
# mk = mask_diastole.unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)


# Create grid of x,y,z coordinates for warping
grid = create_grid(NZ, NY, NX).unsqueeze(dim=0).to(DEVICE)


net = UNet3D(config).to(DEVICE)
net = torch.nn.DataParallel(net, device_ids=[2])
# loss_func = nn.MSELoss(reduction='sum')
# loss_func = nn.BCELoss(reduction='mean')
# loss_func = nn.BCEWithLogitsLoss()
LR = config.getfloat('PARAMETERS', 'LR')
opt = optim.Adam(net.parameters(), lr=LR)

mts = []
mtts = []
# create save directory
saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "CNN")
saveEpochDir = plots.createSubDirectory(saveDir, 'm0tt')
# plots.save_slices(m0.squeeze(), f"m0", saveEpochDir)
plots.save_single_zslices(m0.squeeze(), saveEpochDir, 'm0_gt', max_gray_value=1., color_channel=-1)

fwd_of.reverse()

for e in range(1000):
    net.train()

    mts.append(m0)
    for j in range(len(bwd_of)):
        # Forward mask propagation
        u = bwd_of[j].unsqueeze(dim=0).to(DEVICE)
        mt = net(warp(mts[-1], grid - u))
        # TODO mt = net(myWarp(mts[-1], u))
        mts.append(mt)

    # mtts.append(mts[-1])
    mtts.append(mk)

    for j in range(len(fwd_of)):
        # Backward mask propagation
        u = fwd_of[j].unsqueeze(dim=0).to(DEVICE)
        mtt = net(warp(mtts[-1], grid - u))
        mtts.append(mtt)

    mtts.reverse()
    # plots.save_slices(mtts[0].squeeze(), f"m0tt_{e}", saveEpochDir)
    plots.save_single_zslices(mtts[0].squeeze(), saveEpochDir,
                              f'm0_{e}', max_gray_value=1., color_channel=-1)

    total_loss = 0
    for k in range(len(mtts)):
        # loss = loss_func(mtts[k], mt)
        loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
        total_loss += loss

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    print(total_loss.item())
    mts.clear()
    mtts.clear()
