import torch
import numpy as np
from torch.nn import Module
import sys
import os.path as osp
import os
from warp import Warp
from loss import loss_func_complete
import transforms as T

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.torch_utils as torch_utils


def warp_forward(net: Module, warp, data: torch.Tensor, nsize: tuple, u: torch.Tensor, mt: torch.Tensor):
    data_t = data.reshape(nsize)
    mt = warp(mt.squeeze(), u).reshape(nsize)
    x = torch.cat((data_t, mt), dim=1)
    mt = net(x)
    return mt


def time_propagation(net: Module, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE):
    NZ, NY, NX, NT = vol.shape
    nsize = (1, 1, NZ, NY, NX)
    warp = Warp(config, NZ, NY, NX)
    mts = [m0.reshape(nsize).to(DEVICE)]
    mtts = [mk.reshape(nsize).to(DEVICE)]
    for t in range(len(ff)):
        # Forward mask propagation m0 -> mk
        mt = warp_forward(net, warp, vol[:, :, :, init_ts + t + 1].to(DEVICE), nsize, ff[t].to(DEVICE), mts[-1])
        mts.append(mt)

        # Backward mask propagation mk -> m0
        mtt = warp_forward(net, warp, vol[:, :, :, final_ts - t - 1].to(DEVICE), nsize, bf[t].to(DEVICE), mtts[-1])
        mtts.append(mtt)

    mtts.reverse()
    return (mts, mtts)


def train(net: Module, opt, train_idxs, train_ds, epoch, pbar, config, writer, DEVICE):
    net.train()
    total_train_loss = 0
    normalize = T.Normalize()

    for idx in train_idxs:
        (pname, vol, m0, mk, init_ts, final_ts, ff, bf) = train_ds[idx]
        pbar.set_postfix_str(f'Train P: {pname}, E: {epoch}')
        mts, mtts = time_propagation(net, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE)

        train_loss, *_ = loss_func_complete(mts, mtts)
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        with torch.no_grad():
            total_train_loss += train_loss.item()
            writer.add_image(f'train_{pname}_mkt', torch.swapaxes(normalize(mts[-1]).squeeze(1), 0, 1), dataformats='NCHW')
            writer.add_image(f'train_{pname}_m0tt', torch.swapaxes(normalize(mtts[0]).squeeze(1), 0, 1), dataformats='NCHW')
    return total_train_loss


def validate(net, val_ds, epoch, pbar, config, writer, DEVICE):
    net.eval()
    total_val_loss = 0
    normalize = T.Normalize()

    with torch.no_grad():
        for (pname, vol, m0, mk, init_ts, final_ts, ff, bf) in val_ds:
            pbar.set_postfix_str(f'Val P: {pname}, E: {epoch}')
            mts, mtts = time_propagation(net, vol, m0, mk, init_ts, final_ts, ff, bf, config, DEVICE)

            val_loss, *_ = loss_func_complete(mts, mtts)
            total_val_loss += val_loss.item()
            writer.add_image(f'val_{pname}_mkt', torch.swapaxes(normalize(mts[-1]).squeeze(1), 0, 1), dataformats='NCHW')
            writer.add_image(f'val_{pname}_m0tt', torch.swapaxes(normalize(mtts[0]).squeeze(1), 0, 1), dataformats='NCHW')
    return total_val_loss


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


# ------------------------------------ BATCH OPERATIONS ------------------------------------
class BatchInfo:
    def __init__(self, BS, NZ, NY, NX, dtype, device):
        self.BS = BS
        self.NZ = NZ
        self.NY = NY
        self.NX = NX
        self.dtype = dtype
        self.device = device

    def net_shape(self):
        return (self.BS, 1, self.NZ, self.NY, self.NX)

    def warp_shape(self):
        return (1, self.NZ, self.NY, self.NX)


def create_grid(info: BatchInfo):
    grid = torch_utils.create_grid(info.NZ, info.NY, info.NX).unsqueeze(0).to(info.device)
    grid_b = torch.cat([grid for _ in range(info.BS)], dim=0)
    return grid_b


def train_batch(net, opt, train_loader, pbar, config, DEVICE):
    net.train()
    total_train_loss = 0

    for i, (pnames, vols, m0s, mks, init_ts, final_ts, ff, bf) in enumerate(train_loader):
        pbar.set_postfix_str(f'Train I: {i+1}')

        BS, NZ, NY, NX, _ = vols.shape
        info = BatchInfo(BS, NZ, NY, NX, vols.dtype, DEVICE)

        # max_timesteps = max_ts(ff, info)
        mts = [m0s.reshape(info.net_shape()).to(info.device)]
        mtts = [mks.reshape(info.net_shape()).to(info.device)]
        # warp = Warp(config, info)
        grid = create_grid(info)

        for t in range(ff.shape[1]):
            mt, mtt = batch_warping(ff[:, t, :, :, :, :].to(DEVICE), bf[:, t, :, :, :, :].to(DEVICE), grid, mts[-1], mtts[-1])

            data_t = volume_at_time(vols, init_ts + t + 1).to(DEVICE)
            x = torch.cat((data_t, mt), dim=1)
            x = net(x)
            mts.append(x)

            data_t = volume_at_time(vols, final_ts - t - 1).to(DEVICE)
            x = torch.cat((data_t, mtt), dim=1)
            x = net(x)
            mtts.append(x)

        mtts.reverse()
        train_loss, *_ = loss_func_complete(mts, mtts)
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        with torch.no_grad():
            total_train_loss += train_loss.item()
    return total_train_loss


def batch_warping(ff, bf, grid, mt, mtt):
    # mt_b = torch.zeros(size=info.net_shape(), dtype=info.dtype, device=info.device)
    # mtt_b = torch.zeros(size=info.net_shape(), dtype=info.dtype, device=info.device)

    mt = torch_utils.warp(mt, grid + ff, mode='bilinear')
    mtt = torch_utils.warp(mtt, grid + bf, mode='bilinear')

    # for b in range(info.BS):
    #     uf = ff[b]
    #     ub = bf[b]
    #     maxts = uf.shape[0]
    #     if t < maxts:
    #         # mt = warp((mts[-1][b, :, :, :, :]).squeeze(), uf[t, :, :, :, :].to(info.device)).reshape(info.warp_shape())
    #         # mt_b[b, :, :, :, :] = mt
    #         # mtt = warp((mtts[-1][b, :, :, :, :]).squeeze(), ub[t, :, :, :, :].to(info.device)).reshape(info.warp_shape())
    #         # mtt_b[b, :, :, :, :] = mtt
    #         mt_b = torch_utils.warp(mts[-1], grid + uf, mode='bilinear')
    #         mtt_b = torch_utils.warp(mtts[-1], grid + bf, mode='bilinear')
    #     else:
    #         mt_b[b, :, :, :, :] = torch.zeros(size=info.warp_shape(), dtype=info.dtype, device=info.device)
    #         mtt_b[b, :, :, :, :] = torch.zeros(size=info.warp_shape(), dtype=info.dtype, device=info.device)
    return (mt, mtt)


def max_ts(flow: list, info):
    max_ts = 0
    for b in range(info.BS):
        ts = flow[b].shape[0]
        if ts > max_ts:
            max_ts = ts
    return max_ts


def volume_at_time(vols: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    BS, NZ, NY, NX, NT = vols.shape
    dtype = vols.dtype
    vols_t = torch.zeros(size=(BS, 1, NZ, NY, NX), dtype=dtype)
    for b in range(BS):
        if ts[b] < NT and ts[b] >= 0:
            vols_t[b, :, :, :, :] = vols[b, :, :, :, ts[b]].unsqueeze(0)
        else:
            vols_t[b, :, :, :, :] = torch.zeros(size=(1, NZ, NY, NX), dtype=dtype)
    return vols_t
