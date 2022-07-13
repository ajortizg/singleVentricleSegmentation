import torch
import numpy as np
from torch.nn import Module
import sys
import torch.nn as nn
import os.path as osp
import os
import torch.nn.functional as F
from monai.networks.nets.unet import UNet


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.transforms as T
from cnn.warp import Warp
from cnn.loss import loss_func_complete


def create_model(config, logger):
    num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
    num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
    input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
    features_start = config.getint('PARAMETERS', 'FEATURES_START')
    num_res_units = config.getint('PARAMETERS', 'NUM_RES_UNITS')

    channels = [features_start]
    strides = []
    for _ in range(1, num_layers):
        channels.append(channels[-1] * 2)
        strides.append(2)

    logger.info(f'cnn channels: {channels}')
    logger.info(f'cnn strides: {strides}')
    logger.info(f'cnn res units: {num_res_units}')

    net = UNet(
        spatial_dims=3,
        in_channels=input_channels,
        out_channels=num_classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units
    )

    return net


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


def save_weights(net: Module, epoch: int, every: int, save_dir: str, filename: str):
    if epoch % every == 0:
        torch.save(net, osp.join(save_dir, filename))


def collate_fn(data):
    pnames, vols, m0s, mks, init_ts, final_ts, ff, bf = zip(*data)
    to_tensor = T.ListToTensor()
    vols = to_tensor(vols)
    m0s = to_tensor(m0s)
    mks = to_tensor(mks)
    init_ts = torch.tensor(init_ts)
    final_ts = torch.tensor(final_ts)

    maxts = max_ts(ff)
    BS, NZ, NY, NX = m0s.shape
    fwd_t = torch.zeros(size=(BS, maxts, NZ, NY, NX, 3))
    bwd_t = torch.zeros(size=(BS, maxts, NZ, NY, NX, 3))
    offsets = torch.zeros(BS, dtype=torch.int)
    for b in range(BS):
        diff_t = (int)(maxts - ff[b].shape[0])
        offsets[b] = diff_t
        if diff_t != 0:
            zeros = torch.zeros(size=(diff_t, NZ, NY, NX, 3))
            fwd_t[b, :, :, :, :, :] = torch.cat((ff[b], zeros), dim=0)
            bwd_t[b, :, :, :, :, :] = torch.cat((bf[b], zeros), dim=0)
        else:
            fwd_t[b, :, :, :, :, :] = ff[b]
            bwd_t[b, :, :, :, :, :] = bf[b]

    return (pnames, vols.unsqueeze_(1), m0s.unsqueeze_(1), mks.unsqueeze_(1), init_ts, final_ts, fwd_t, bwd_t, offsets)


def max_ts(ff):
    BS = len(ff)
    maxts = 0
    for b in range(BS):
        t = ff[b].shape[0]
        if t > maxts:
            maxts = t
    return maxts


# def warp_forward(net: Module, warp, data: torch.Tensor, nsize: tuple, u: torch.Tensor, mt: torch.Tensor):
#     data_t = data.reshape(nsize)
#     mt = warp(mt.squeeze(), u).unsqueeze(0).unsqueeze(0)
#     x = torch.cat((data_t, mt), dim=1)
#     x = net(x)
#     x = torch.sigmoid(x)
#     return x


# def time_propagation(net: Module, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE):
#     NZ, NY, NX, NT = vol.shape
#     nsize = (1, 1, NZ, NY, NX)
#     warp = Warp(config, NZ, NY, NX)
#     mts = [m0.reshape(nsize).to(DEVICE)]
#     mtts = [mk.reshape(nsize).to(DEVICE)]
#     ts = ff.shape[0]
#     for t in range(ts):
#         # Forward mask propagation m0 -> mk
#         mt = warp_forward(net, warp, vol[:, :, :, init_ts + t + 1].to(DEVICE), nsize, ff[t, :, :, :, :].to(DEVICE), mts[-1])
#         mts.append(mt)

#         # Backward mask propagation mk -> m0
#         mtt = warp_forward(net, warp, vol[:, :, :, final_ts - t - 1].to(DEVICE), nsize, bf[t, :, :, :, :].to(DEVICE), mtts[-1])
#         mtts.append(mtt)

#     mtts.reverse()
#     return (mts, mtts)


# def train(net: Module, opt, train_idxs, train_ds, epoch, pbar, config, writer, DEVICE):
#     net.train()
#     total_train_loss = 0
#     # normalize = T.Normalize()

#     for idx in train_idxs:
#         (pname, vol, m0, mk, init_ts, final_ts, ff, bf) = train_ds[idx]
#         pbar.set_postfix_str(f'Train P: {pname}, E: {epoch}')
#         mts, mtts = time_propagation(net, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE)

#         train_loss, *_ = loss_func_complete(mts, mtts)
#         opt.zero_grad()
#         train_loss.backward()
#         opt.step()

#         with torch.no_grad():
#             total_train_loss += train_loss.item()
#             # writer.add_image(f'train_{pname}_mkt', torch.swapaxes(normalize(mts[-1]).squeeze(1), 0, 1), dataformats='NCHW')
#             # writer.add_image(f'train_{pname}_m0tt', torch.swapaxes(normalize(mtts[0]).squeeze(1), 0, 1), dataformats='NCHW')
#     return total_train_loss


# def validate(net, val_ds, epoch, pbar, config, writer, DEVICE):
#     net.eval()
#     total_val_loss = 0
#     # normalize = T.Normalize()

#     with torch.no_grad():
#         for (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in val_ds:
#             pbar.set_postfix_str(f'Val P: {pname}, E: {epoch}')
#             mts, mtts = time_propagation(net, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE)

#             val_loss, *_ = loss_func_complete(mts, mtts)
#             total_val_loss += val_loss.item()
#             # writer.add_image(f'val_{pname}_mkt', torch.swapaxes(normalize(mts[-1]).squeeze(1), 0, 1), dataformats='NCHW')
#             # writer.add_image(f'val_{pname}_m0tt', torch.swapaxes(normalize(mtts[0]).squeeze(1), 0, 1), dataformats='NCHW')
#     return total_val_loss


# def train_batch(net, opt, train_loader, pbar, config, DEVICE):
#     net.train()
#     total_train_loss = 0

#     for i, (pnames, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets) in enumerate(train_loader):
#         pbar.set_postfix_str(f'Train I: {i+1}')
#         mts, mtts = time_popagation_batch(net, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets, config, DEVICE)

#         train_loss = loss_func_batch(mts, mtts, offsets, loss_lambda)
#         opt.zero_grad()
#         train_loss.backward()
#         opt.step()

#         with torch.no_grad():
#             total_train_loss += train_loss.item()
#     return total_train_loss


# def val_batch(net, val_loader, pbar, config, DEVICE):
#     net.eval()
#     total_val_loss = 0

#     with torch.no_grad():
#         for i, (pnames, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets) in enumerate(val_loader):
#             pbar.set_postfix_str(f'Val I: {i+1}')
#             mts, mtts = time_popagation_batch(net, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets, config, DEVICE)

#             val_loss = loss_func_batch(mts, mtts, offsets, loss_lambda)
#             total_val_loss += val_loss.item()
#     return total_val_loss


# def time_popagation_batch(net, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets, config, DEVICE):
#     mts = [m0s.to(DEVICE)]
#     mtts = [mks.to(DEVICE)]
#     flow_times = ff.shape[1]
#     diff_t = flow_times - offsets
#     NZ, NY, NX = m0s.shape[2:]
#     warp = WarpCNN(config, NZ, NY, NX)

#     for t in range(flow_times):
#         # Forward propagation m0 -> mk
#         mt = warp(mts[-1], ff[:, t, :, :, :, :].to(DEVICE))
#         data_t = select_volume_fwd(vols, init_ts, final_ts, t, diff_t).to(DEVICE)
#         x = torch.cat((data_t, mt), dim=1)
#         x = net(x)
#         mts.append(x)

#         # Backward propagation mk -> m0
#         mtt = warp(mtts[-1], bf[:, t, :, :, :, :].to(DEVICE))
#         data_t = select_volume_bwd(vols, init_ts, final_ts, t, diff_t).to(DEVICE)
#         x = torch.cat((data_t, mtt), dim=1)
#         x = net(x)
#         mtts.append(x)

#     mtts.reverse()
#     return (mts, mtts)


# def select_volume_fwd(vols: torch.Tensor, init_ts: torch.Tensor, final_ts: torch.Tensor, cur_t: int, diff_t: torch.Tensor) -> torch.Tensor:
#     ts = init_ts + cur_t + 1

#     BS, CH, NZ, NY, NX, NT = vols.shape
#     dtype = vols.dtype
#     vols_t = torch.zeros(size=(BS, CH, NZ, NY, NX), dtype=dtype)

#     for b in range(BS):
#         if cur_t >= diff_t[b]:
#             vols_t[b, :, :, :, :] = vols[b, :, :, :, :, final_ts[b]]
#             # print(f'correct: b: {b} - {cur_t} - {final_ts[b]}')
#         else:
#             vols_t[b, :, :, :, :] = vols[b, :, :, :, :, ts[b]]
#             # print(f'normal: b: {b} - {cur_t} - {ts[b]}')

#     return vols_t


# def select_volume_bwd(vols: torch.Tensor, init_ts: torch.Tensor, final_ts: torch.Tensor, cur_t: int, diff_t: torch.Tensor) -> torch.Tensor:
#     ts = final_ts - cur_t - 1

#     BS, CH, NZ, NY, NX, NT = vols.shape
#     dtype = vols.dtype
#     vols_t = torch.zeros(size=(BS, CH, NZ, NY, NX), dtype=dtype)

#     for b in range(BS):
#         if cur_t >= diff_t[b]:
#             vols_t[b, :, :, :, :] = vols[b, :, :, :, :, init_ts[b]]
#             # print(f'correct: b: {b} - {cur_t} - {init_ts[b]}')
#         else:
#             vols_t[b, :, :, :, :] = vols[b, :, :, :, :, ts[b]]
#             # print(f'normal: b: {b} - {cur_t} - {ts[b]}')

#     return vols_t
