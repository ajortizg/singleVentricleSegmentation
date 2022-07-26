import os.path as osp
import sys
import torch
from torch import nn

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utils.transforms as T
from cnn.warp import WarpCNN
from cnn import metrics
from loss import DiceBCELoss


class Trainer:
    def __init__(self, net, pbar, config, device):
        self.net = net
        self.pbar = pbar
        self.config = config
        self.device = device
        self.loss_lambda = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')
        loss_fn_type = config.get('PARAMETERS', 'LOSS_FN')

        reduction = config.get('PARAMETERS', 'LOSS_REDUCTION')
        self.loss_fn = None
        self.compute_l3 = True
        if loss_fn_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_fn_type == 'huber':
            huber_delta = config.getfloat('PARAMETERS', 'HUBER_DELTA')
            self.loss_fn = nn.HuberLoss(reduction=reduction, delta=huber_delta)
        elif loss_fn_type == 'bce':
            self.loss_fn = nn.BCELoss(reduction=reduction)
            self.compute_l3 = False
        elif loss_fn_type == 'bce_dice':
            alpha = config.getfloat('PARAMETERS', 'LOSS_ALPHA')
            self.loss_fn = DiceBCELoss(alpha=alpha, reduction=reduction)
            self.compute_l3 = False
        else:
            print('Unknown loss function: ' + loss_fn_type)
            sys.exit()

    def train_epoch(self, train_loader, opt):
        self.net.train()
        total_train_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        total_l3_loss = 0
        total_mean_acc = 0

        for i, (pnames, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets) in enumerate(train_loader):
            self.pbar.set_postfix_str(f'Train: {i+1}/{len(train_loader)}')
            imgs4d = imgs4d.to(self.device)
            m0s = m0s.to(self.device)
            mks = mks.to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)
            offsets = offsets.to(torch.long)
            BS = offsets.shape[0]
            batch_indices = torch.arange(BS)

            mts, mtts = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
            train_loss, l1, l2, l3 = self.compute_loss(mts, mtts, offsets, batch_indices)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                total_train_loss += train_loss.item()
                total_l1_loss += l1
                total_l2_loss += l2
                total_l3_loss += l3
                total_mean_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
        return (total_train_loss, total_l1_loss, total_l2_loss, total_l3_loss, total_mean_acc)

    def val_epoch(self, val_loader):
        self.net.eval()
        total_val_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        total_l3_loss = 0
        total_mean_acc = 0

        with torch.no_grad():
            for i, (pnames, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets) in enumerate(val_loader):
                self.pbar.set_postfix_str(f'Val: {i+1}/{len(val_loader)}')
                imgs4d = imgs4d.to(self.device)
                m0s = m0s.to(self.device)
                mks = mks.to(self.device)
                ff = ff.to(self.device)
                bf = bf.to(self.device)
                offsets = offsets.to(torch.long)
                BS = offsets.shape[0]
                batch_indices = torch.arange(BS)

                mts, mtts = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
                val_loss, l1, l2, l3 = self.compute_loss(mts, mtts, offsets, batch_indices)

                total_val_loss += val_loss.item()
                total_l1_loss += l1
                total_l2_loss += l2
                total_l3_loss += l3
                total_mean_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
        return (total_val_loss, total_l1_loss, total_l2_loss, total_l3_loss, total_mean_acc)

    def train_patient(self, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets, opt):
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)

        self.net.train()
        mts, mtts = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
        train_loss, l1, l2, l3 = self.compute_loss(mts, mtts, offsets, batch_indices)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        with torch.no_grad():
            mean_acc = self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
        return (train_loss.item(), l1, l2, l3, mean_acc)

    def val_patient(self, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, offsets):
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)
        self.net.eval()
        
        with torch.no_grad():
            mts, mtts = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
            val_loss, l1, l2, l3 = self.compute_loss(mts, mtts, offsets, batch_indices)
            mean_acc = self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
        return (val_loss.item(), l1, l2, l3, mean_acc)

    def compute_dice_acc(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)
        
        mk = mtts[-1]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        acc0 = metrics.dice(m0tt, m0)
        acck = metrics.dice(mkt, mk)
        dc = 0.5 * (acc0 + acck)
        return dc

    def time_popagation(self, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices):
        BS, timesteps, NZ, NY, NX, _ = ff.shape
        dtype = m0s.dtype
        warp = WarpCNN(self.config, NZ, NY, NX)
        mts = torch.empty(size=(timesteps + 1, BS, 1, NZ, NY, NX), dtype=dtype, device=self.device)
        mts[0] = m0s
        mtts = torch.empty_like(mts)
        mtts[-1] = mks

        for t in range(timesteps):
            # Forward propagation m0 -> mk
            mt = warp(mts[t], ff[:, t, :, :, :, :])
            x = torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_fwd[t + 1][batch_indices]], mt), dim=1)
            mts[t + 1] = self.net(x)

            # Backward propagation mk -> m0
            mtt = warp(mtts[timesteps - t], bf[:, t, :, :, :, :])
            x = torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_bwd[t + 1][batch_indices]], mtt), dim=1)
            mtts[timesteps - t - 1] = self.net(x)
        return (mts, mtts)

    def compute_loss(self, mts: torch.Tensor, mtts: torch.Tensor, offsets: torch.Tensor, batch_indices: torch.Tensor):
        BS = mts.shape[1]

        # compute l1
        m0 = mts[0, batch_indices]
        # m0tt = mtts[0]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        l1 = self.loss_fn(m0tt, m0)

        # compute l2
        mk = mtts[-1]
        # mkt = mts[-1]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        l2 = self.loss_fn(mkt, mk)

        if not self.compute_l3:
            # For bce and bcd_dice loss
            l3 = torch.tensor([0.0], device=self.device)
            return (l1 + l2, l1.item(), l2.item(), l3.item())
        else:
            # compute l3
            timesteps = mts.shape[0]
            l3 = 0
            for b in range(BS):
                mt = mts[1:timesteps - offsets[b] - 1, b, :, :, :, :]
                mtt = mtts[1 + offsets[b]:-1, b, :, :, :, :]
                # l3 += self.loss_fn(mt, mtt) / mt.shape[0]
                l3 += self.loss_fn(mt, mtt)

            # l1 = l1 / BS
            # l2 = l2 / BS
            # l3 = self.loss_lambda * l3
            l3 = self.loss_lambda * l3 / BS
            total_loss = l1 + l2 + l3
        return (total_loss, l1.item(), l2.item(), l3.item())
