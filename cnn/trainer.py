import os.path as osp
import sys
import torch
from torch import nn
from monai.metrics.meandice import compute_meandice
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.warp import WarpCNN


class Trainer:
    def __init__(self, net, pbar, config, device, writer, display_prob=0.2):
        self.net = net
        self.pbar = pbar
        self.config = config
        self.device = device
        self.writer = writer
        self.display_prob = display_prob
        self.loss_lambda = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')
        loss_fn_type = config.get('PARAMETERS', 'LOSS_FN')
        self.penalization = config.get('PARAMETERS', 'LOSS_PENALIZATION')
        self.mu = config.getfloat('PARAMETERS', 'LOSS_PENALIZATION_MU')
        self.reduction = config.get('PARAMETERS', 'LOSS_REDUCTION')
        self.loss_fn = None

        if loss_fn_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=self.reduction)
        elif loss_fn_type == 'huber':
            huber_delta = config.getfloat('PARAMETERS', 'HUBER_DELTA')
            self.loss_fn = nn.HuberLoss(reduction=self.reduction, delta=huber_delta)
        else:
            print('Unknown loss function: ' + loss_fn_type)
            sys.exit()

    def train_epoch(self, train_loader, opt):
        self.net.train()
        total_train_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        total_l3_loss = 0
        total_l4_loss = 0
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

            mts, mtts, mhs, mhhs = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
            train_loss, l1, l2, l3, l4 = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                total_train_loss += train_loss.item()
                total_l1_loss += l1
                total_l2_loss += l2
                total_l3_loss += l3
                total_l4_loss += l4
                total_mean_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
                if np.random.rand() < self.display_prob:
                    self.plot_imgs(mts, mtts, offsets, batch_indices, 'train')
        return (total_train_loss, total_l1_loss, total_l2_loss, total_l3_loss, total_l4_loss, total_mean_acc)

    def val_epoch(self, val_loader):
        self.net.eval()
        total_val_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        total_l3_loss = 0
        total_l4_loss = 0
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

                mts, mtts, mhs, mhhs = self.time_popagation(imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices)
                val_loss, l1, l2, l3, l4 = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

                total_val_loss += val_loss.item()
                total_l1_loss += l1
                total_l2_loss += l2
                total_l3_loss += l3
                total_l4_loss += l4
                total_mean_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
                if np.random.rand() < self.display_prob:
                    self.plot_imgs(mts, mtts, offsets, batch_indices, 'val')
        return (total_val_loss, total_l1_loss, total_l2_loss, total_l3_loss, total_l4_loss, total_mean_acc)

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

    def time_popagation(self, imgs4d, m0s, mks, list_times_fwd, list_times_bwd, ff, bf, batch_indices):
        BS, timesteps, NZ, NY, NX, _ = ff.shape
        dtype = m0s.dtype
        warp = WarpCNN(self.config, NZ, NY, NX)
        mts = torch.empty(size=(timesteps + 1, BS, 1, NZ, NY, NX), dtype=dtype, device=self.device)
        mts[0] = m0s
        mtts = torch.empty_like(mts)
        mtts[-1] = mks

        if self.penalization:
            mhs = torch.empty(size=(timesteps, BS, 1, NZ, NY, NX), dtype=dtype, device=self.device)
            mhhs = torch.empty_like(mhs)
        else:
            mhs = mhhs = None

        for t in range(timesteps):
            # Forward propagation m0 -> mk
            mt = warp(mts[t], ff[:, t, :, :, :, :])
            x = torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_fwd[t + 1][batch_indices]], mt), dim=1)
            mts[t + 1], mh = self.net(x)

            # Backward propagation mk -> m0
            mtt = warp(mtts[timesteps - t], bf[:, t, :, :, :, :])
            x = torch.cat((imgs4d[batch_indices, :, :, :, :, list_times_bwd[t + 1][batch_indices]], mtt), dim=1)
            mtts[timesteps - t - 1], mhh = self.net(x)

            if self.penalization:
                mhs[t] = mh
                mhhs[t] = mhh
        return (mts, mtts, mhs, mhhs)

    def compute_dice_acc(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        acc0 = compute_meandice(m0tt, m0).mean()
        acck = compute_meandice(mkt, mk).mean()
        dc = 0.5 * (acc0 + acck)

        return dc

    def plot_imgs(self, mts, mtts, offsets, batch_indices, tag):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        for b in range(len(offsets)):
            m0_b = m0[b]
            m0_b.swapaxes_(0, 1)
            m0tt_b = m0tt[b]
            m0tt_b.swapaxes_(0, 1)
            m0_e = torch.abs(m0_b - m0tt_b)
            self.writer.add_images(f'{tag}/m0_b{b}/gt', m0_b)
            self.writer.add_images(f'{tag}/m0_b{b}/est', m0tt_b)
            self.writer.add_images(f'{tag}/m0_b{b}/error', m0_e)

            mk_b = mk[b]
            mk_b.swapaxes_(0, 1)
            mkt_b = mkt[b]
            mkt_b.swapaxes_(0, 1)
            mk_e = torch.abs(mk_b - mkt_b)
            self.writer.add_images(f'{tag}/mk_b{b}/gt', mk_b)
            self.writer.add_images(f'{tag}/mk_b{b}/est', mkt_b)
            self.writer.add_images(f'{tag}/mk_b{b}/error', mk_e)

    def compute_loss(self, mts, mtts, offsets, batch_indices, mhs=None, mhhs=None):
        BS = mts.shape[1]

        # compute l1
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        l1 = self.loss_fn(m0tt, m0)

        # compute l2
        mk = mtts[-1]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        l2 = self.loss_fn(mkt, mk)

        # compute l3
        timesteps = mts.shape[0]
        l3 = 0
        for b in range(BS):
            mt = mts[1:timesteps - offsets[b] - 1, b, :, :, :, :]
            mtt = mtts[1 + offsets[b]:-1, b, :, :, :, :]
            if self.reduction == 'sum':
                l3 += self.loss_fn(mt, mtt) / mt.shape[0]
            else:
                l3 += self.loss_fn(mt, mtt)

        # compute l4 - penalization term
        l4 = 0.0
        if self.penalization:
            timesteps = mhs.shape[0]
            for b in range(BS):
                mh = mhs[0:timesteps - offsets[b], b]
                mhh = mhhs[0:timesteps - offsets[b], b]
                l4 += torch.norm(mh)**2 + torch.norm(mhh)**2
            # l4 = self.mu * (torch.norm(mhs)**2 + torch.norm(mhhs)**2)
            l4 = self.mu * l4 / BS

        if self.reduction == 'sum':
            l1 = l1 / BS
            l2 = l2 / BS
        l3 = self.loss_lambda * l3 / BS
        total_loss = l1 + l2 + l3 + l4
        return (total_loss, l1.item(), l2.item(), l3.item(), l4.item())
