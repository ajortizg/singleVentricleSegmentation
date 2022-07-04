import os.path as osp
import sys
import torch

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.transforms as T
from cnn.warp import WarpCNN, Warp
from cnn.loss import loss_func_batch


class Trainer:
    def __init__(self, net, pbar, config, device):
        self.net = net
        self.pbar = pbar
        self.config = config
        self.device = device

        self.loss_lambda = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')

    def train_epoch(self, train_loader, opt):
        self.net.train()
        total_train_loss = 0

        for i, (pnames, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets) in enumerate(train_loader):
            self.pbar.set_postfix_str(f'Train I: {i+1}')
            mts, mtts = self.time_popagation_batch(vols, m0s, mks, init_ts, final_ts, ff, bf, offsets)

            train_loss = loss_func_batch(mts, mtts, offsets, self.loss_lambda)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                total_train_loss += train_loss.item()
        return total_train_loss

    def val_epoch(self, val_loader):
        self.net.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i, (pnames, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets) in enumerate(val_loader):
                self.pbar.set_postfix_str(f'Val I: {i+1}')
                mts, mtts = self.time_popagation_batch(vols, m0s, mks, init_ts, final_ts, ff, bf, offsets)

                val_loss = loss_func_batch(mts, mtts, offsets, self.loss_lambda)
                total_val_loss += val_loss.item()
        return total_val_loss

    def time_popagation_batch(self, vols, m0s, mks, init_ts, final_ts, ff, bf, offsets):
        mts = [m0s.to(self.device)]
        mtts = [mks.to(self.device)]
        flow_times = ff.shape[1]
        diff_t = flow_times - offsets
        NZ, NY, NX = m0s.shape[2:]
        warp = WarpCNN(self.config, NZ, NY, NX)

        for t in range(flow_times):
            # Forward propagation m0 -> mk
            mt = warp(mts[-1], ff[:, t, :, :, :, :].to(self.device))
            data_t = self.select_volume_fwd(vols, init_ts, final_ts, t, diff_t).to(self.device)
            x = torch.cat((data_t, mt), dim=1)
            x = self.net(x)
            mts.append(x)

            # Backward propagation mk -> m0
            mtt = warp(mtts[-1], bf[:, t, :, :, :, :].to(self.device))
            data_t = self.select_volume_bwd(vols, init_ts, final_ts, t, diff_t).to(self.device)
            x = torch.cat((data_t, mtt), dim=1)
            x = self.net(x)
            mtts.append(x)

        mtts.reverse()
        to_tensor = T.ListToTensor()
        return (to_tensor(mts), to_tensor(mtts))

    def select_volume_fwd(self, vols: torch.Tensor, init_ts: torch.Tensor, final_ts: torch.Tensor, cur_t: int, diff_t: torch.Tensor) -> torch.Tensor:
        ts = init_ts + cur_t + 1

        BS, CH, NZ, NY, NX, NT = vols.shape
        dtype = vols.dtype
        vols_t = torch.zeros(size=(BS, CH, NZ, NY, NX), dtype=dtype)

        for b in range(BS):
            if cur_t >= diff_t[b]:
                vols_t[b, :, :, :, :] = vols[b, :, :, :, :, final_ts[b]]
                # print(f'correct: b: {b} - {cur_t} - {final_ts[b]}')
            else:
                vols_t[b, :, :, :, :] = vols[b, :, :, :, :, ts[b]]
                # print(f'normal: b: {b} - {cur_t} - {ts[b]}')

        return vols_t

    def select_volume_bwd(self, vols: torch.Tensor, init_ts: torch.Tensor, final_ts: torch.Tensor, cur_t: int, diff_t: torch.Tensor) -> torch.Tensor:
        ts = final_ts - cur_t - 1

        BS, CH, NZ, NY, NX, NT = vols.shape
        dtype = vols.dtype
        vols_t = torch.zeros(size=(BS, CH, NZ, NY, NX), dtype=dtype)

        for b in range(BS):
            if cur_t >= diff_t[b]:
                vols_t[b, :, :, :, :] = vols[b, :, :, :, :, init_ts[b]]
                # print(f'correct: b: {b} - {cur_t} - {init_ts[b]}')
            else:
                vols_t[b, :, :, :, :] = vols[b, :, :, :, :, ts[b]]
                # print(f'normal: b: {b} - {cur_t} - {ts[b]}')

        return vols_t
