import torch
import numpy as np
from torch.nn import Module
import sys
import os.path as osp
import os


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots


def propagate_batch(net, warp, data, u, mt):
    # data shape:   BS, NZ, NY, NX, 4
    # u shape:      BS, NZ, NY, NX, 3
    # mt shape:     BS, 1, NZ, NY, NX
    # nsize:        BS, 1, NZ, NY, NX
    BS, CH, NZ, NY, NX = mt.shape
    dtype = mt.dtype
    device = mt.device
    datac = torch.zeros((BS, NZ, NY, NX), dtype=dtype, device=device)
    mts = torch.zeros((BS, NZ, NY, NX), device=device, dtype=dtype)

    for batch in range(BS):
        mts[batch, :, :, :] = warp(mt.squeeze()[batch, :, :, :], u[batch, :, :, :, :])
        datac[batch, :, :, :] = data[batch, :, :, :, batch].to(device)

    nsize = (BS, CH, NZ, NY, NX)
    x = torch.cat((datac.reshape(nsize), mts.reshape(nsize)), dim=1)
    y = net(x)
    return y


def propagate(net: Module, warp, data: torch.Tensor, nsize: tuple, u: torch.Tensor, mt: torch.Tensor):
    data_t = data.reshape(nsize)
    mt = warp(mt.squeeze(), u).reshape(nsize)
    x = torch.cat((data_t, mt), dim=1)
    mt = net(x)
    return mt


def save_train_val_patients(ds, train_idxs, val_idxs, save_dir: str):
    filepath = osp.join(save_dir, 'train_patients.txt')
    with open(filepath, 'w') as pfile:
        for idx in train_idxs:
            pfile.write(ds.get_patient_name(idx) + '\n')
    pfile.close()

    filepath = osp.join(save_dir, 'val_patients.txt')
    with open(filepath, 'w') as pfile:
        for idx in val_idxs:
            pfile.write(ds.get_patient_name(idx) + '\n')
    pfile.close()


def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(net: Module, save_dir: str, filename: str):
    filepath = osp.join(save_dir, filename)
    total_params = 0

    modules = [module for module in net.modules()]
    params = [param for param in net.parameters()]
    with open(filepath, 'w') as mfile:
        for idx, m in enumerate(modules):
            mfile.write(f'{idx} -> {m}\n')
        for p in params:
            total_params += p.numel()
        mfile.write(f'\nTotal parameters: {total_params}')
    mfile.close()
