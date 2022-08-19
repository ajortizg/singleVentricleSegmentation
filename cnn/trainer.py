import os.path as osp
import sys
import torch
from torch import nn
from monai.metrics.meandice import compute_meandice
import numpy as np
import os
from monai.visualize import plot_2d_or_3d_image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.warp import WarpCNN


__all__ = ['Trainer']


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
        self.penalization = config.getboolean('PARAMETERS', 'LOSS_PENALIZATION')
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

    def train_epoch(self, train_loader, opt, cnn=True):
        self.net.train()
        total_loss = (0.0, 0.0, 0.0, 0.0, 0.0)
        total_acc = 0

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(train_loader):
            self.pbar.set_postfix_str(f'Train: {i+1}/{len(train_loader)}')
            img4d = img4d.to(self.device)
            m0 = m0.to(self.device)
            mk = mk.to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)
            offsets = offsets.to(torch.long)
            batch_indices = torch.arange(offsets.shape[0])

            mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
            loss = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

            opt.zero_grad()
            loss[0].backward()
            opt.step()

            with torch.no_grad():
                total_loss = tuple(tl + l.item() for tl, l in zip(total_loss, loss))
                total_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
                if np.random.rand() < self.display_prob:
                    self.plot_imgs(mts, mtts, offsets, batch_indices, 'train')
        return (*total_loss, total_acc)

    def val_epoch(self, val_loader, cnn=True):
        self.net.eval()
        total_loss = (0.0, 0.0, 0.0, 0.0, 0.0)
        total_acc = 0

        with torch.no_grad():
            for i, (pnames, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(val_loader):
                self.pbar.set_postfix_str(f'Val: {i+1}/{len(val_loader)}')
                img4d = img4d.to(self.device)
                m0 = m0.to(self.device)
                mk = mk.to(self.device)
                ff = ff.to(self.device)
                bf = bf.to(self.device)
                offsets = offsets.to(torch.long)
                batch_indices = torch.arange(offsets.shape[0])

                mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
                loss = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

                total_loss = tuple(tl + l.item() for tl, l in zip(total_loss, loss))
                total_acc += self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
                if np.random.rand() < self.display_prob:
                    self.plot_imgs(mts, mtts, offsets, batch_indices, 'val')
        return (*total_loss, total_acc)

    @torch.no_grad()
    def test_epoch(self, test_loader, test_ds):
        self.net.eval()
        total_acc = 0

        for i, (pnames, img4d, _, _, masks, times_fwd, times_bwd, ff, bf, _) in enumerate(test_loader):
            self.pbar.set_postfix_str(f'Test: {i+1}/{len(test_loader)}')
            times_fwd, times_bwd = test_ds.create_timeline(times_fwd[0], times_bwd[0], masks.shape[-1])
            img4d = img4d.to(self.device)
            masks = masks.to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)

            BS, NZ, NY, NX, _, timesteps = ff.shape
            warp = WarpCNN(self.config, NZ, NY, NX)
            mts = torch.empty_like(masks)
            mts[..., times_fwd[0]] = masks[..., times_fwd[0]]
            mtts = torch.empty_like(masks)
            mtts[..., times_bwd[0]] = masks[..., times_bwd[0]]

            for t in range(timesteps - 1):
                # Forward mask propagation
                mt = warp(mts[..., times_fwd[t]], ff[..., t])
                x = torch.cat((img4d[..., times_fwd[t + 1]], mt), dim=1)
                mts[..., times_fwd[t + 1]], _ = self.net(x)

                # Backward mask propagation
                mtt = warp(mtts[..., times_bwd[t]], bf[..., t])
                x = torch.cat((img4d[..., times_bwd[t + 1]], mtt), dim=1)
                mtts[..., times_bwd[t + 1]], _ = self.net(x)

            # Compute forward accuracy
            mts = mts.swapaxes(0, -1).squeeze(-1)
            masks = masks.swapaxes(0, -1).squeeze(-1)
            mts = torch.where(mts > 0.5, 1.0, 0.0)
            acc_fwd = compute_meandice(mts, masks).mean().item()

            # Compute backward accuracy
            mtts = mtts.swapaxes(0, -1).squeeze(-1)
            mtts = torch.where(mtts > 0.5, 1.0, 0.0)
            acc_bwd = compute_meandice(mtts, masks).mean().item()

            # Mean acc
            acc = 0.5 * (acc_fwd + acc_bwd)
            total_acc += acc
            if np.random.rand() < self.display_prob:
                self.plot_test_imgs(mts, masks, 'test')
        return total_acc

        #         BS, NZ, NY, NX, _, timesteps = ff.shape
        #         warp = WarpCNN(self.config, NZ, NY, NX)
        #         mts = torch.empty(size=(timesteps, BS, 1, NZ, NY, NX), dtype=img4d.dtype, device=self.device)
        #         mts[0] = masks[..., times_fwd[0]]
        #         masks_gt = torch.empty_like(mts)
        #         masks_gt[0] = masks[..., times_fwd[0]]

        #         for t in range(timesteps - 1):
        #             # Forward mask propagation
        #             mt = warp(mts[t], ff[..., t])
        #             x = torch.cat((img4d[..., times_fwd[t + 1]], mt), dim=1)
        #             mts[t + 1], _ = self.net(x)
        #             masks_gt[t + 1] = masks[..., times_fwd[t + 1]]

        #         # Compute accuracy
        #         mts.swapaxes_(0, 1).squeeze_(0)
        #         mts = torch.where(mts > 0.5, 1.0, 0.0)
        #         masks_gt.swapaxes_(0, 1).squeeze_(0)
        #         acc = compute_meandice(mts, masks_gt).mean()
        #         total_acc += acc.item()

        #         if np.random.rand() < self.display_prob:
        #             self.plot_test_imgs(mts, masks_gt, 'test')
        # return total_acc

    def time_popagation(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices, cnn=True):
        BS, NZ, NY, NX, CH, timesteps = ff.shape
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
            mt = warp(mts[t], ff[..., t])
            if cnn:
                x = torch.cat((imgs4d[batch_indices, ..., times_fwd[t + 1][batch_indices]], mt), dim=1)
                mts[t + 1], mh = self.net(x)
            else:
                mts[t + 1] = mt

            # Backward propagation mk -> m0
            mtt = warp(mtts[timesteps - t], bf[..., t])
            if cnn:
                x = torch.cat((imgs4d[batch_indices, ..., times_bwd[t + 1][batch_indices]], mtt), dim=1)
                mtts[timesteps - t - 1], mhh = self.net(x)
            else:
                mtts[timesteps - t - 1] = mtt

            if self.penalization:
                mhs[t] = mh
                mhhs[t] = mhh
        return (mts, mtts, mhs, mhhs)

    def compute_dice_acc(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        acc0 = compute_meandice(m0tt, m0).mean()
        acck = compute_meandice(mkt, mk).mean()
        dc = 0.5 * (acc0 + acck)
        return dc

    def plot_test_imgs(self, mts, gts, tag):
        b = np.random.randint(mts.shape[0])
        mt = mts[b]
        gt = gts[b]
        mt = mt.swapaxes_(0, 1)
        gt = gt.swapaxes_(0, 1)
        error = torch.abs(gt - mt)
        self.writer.add_images(f'{tag}/gt', gt)
        self.writer.add_images(f'{tag}/est', mt)
        self.writer.add_images(f'{tag}/error', error)

    def plot_imgs(self, mts, mtts, offsets, batch_indices, tag):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        # m0 = m0.permute(0, 1, 3, 4, 2)
        # m0tt = m0tt.permute(0, 1, 3, 4, 2)
        # m0_e = torch.abs(m0 - m0tt)

        # mk = mk.permute(0, 1, 3, 4, 2)
        # mkt = mkt.permute(0, 1, 3, 4, 2)
        # mk_e = torch.abs(mk - mkt)

        # plot_2d_or_3d_image(m0, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/m0/gt')
        # plot_2d_or_3d_image(m0tt, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/m0/pred')
        # plot_2d_or_3d_image(m0_e, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/m0/error')

        # plot_2d_or_3d_image(mk, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/mk/gt')
        # plot_2d_or_3d_image(mkt, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/mk/pred')
        # plot_2d_or_3d_image(mk_e, step=0, writer=self.writer, frame_dim=-1, tag=f'{tag}/mk/error')

        # take batch randomly
        b = np.random.randint(len(offsets))
        m0_b = m0[b]
        m0tt_b = m0tt[b]
        m0_b.swapaxes_(0, 1)
        m0tt_b.swapaxes_(0, 1)
        m0_e = torch.abs(m0_b - m0tt_b)
        self.writer.add_images(f'{tag}/m0_b{b}/gt', m0_b)
        self.writer.add_images(f'{tag}/m0_b{b}/est', m0tt_b)
        self.writer.add_images(f'{tag}/m0_b{b}/error', m0_e)

        mk_b = mk[b]
        mkt_b = mkt[b]
        mk_b.swapaxes_(0, 1)
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
        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        l2 = self.loss_fn(mkt, mk)

        # compute l3
        timesteps = mts.shape[0]
        l3 = 0
        for b in range(BS):
            mt = mts[1:timesteps - offsets[b] - 1, b]
            mtt = mtts[1 + offsets[b]:-1, b]
            if self.reduction == 'sum':
                l3 += self.loss_fn(mt, mtt) / mt.shape[0]
            else:
                l3 += self.loss_fn(mt, mtt)

        # compute l4 - penalization term
        l4 = torch.tensor([0.0], dtype=l1.dtype, device=l1.device)
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
        return (total_loss, l1, l2, l3, l4)

    def train_patient(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, offsets, opt):
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)

        self.net.train()
        mts, mtts, mhs, mhhs = self.time_popagation(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices)
        loss = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

        opt.zero_grad()
        loss[0].backward()
        opt.step()

        with torch.no_grad():
            acc = self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
            loss = tuple(l.item() for l in loss)
            if np.random.rand() < self.display_prob:
                self.plot_imgs(mts, mtts, offsets, batch_indices, 'train')
        return (*loss, acc)

    @torch.no_grad()
    def val_patient(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, offsets, cnn):
        self.net.eval()
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)

        mts, mtts, *_ = self.time_popagation(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
        # loss = self.compute_loss(mts, mtts, offsets, batch_indices)
        acc = self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()
        # loss = tuple(l.item() for l in loss)
        return acc, mts, mtts

    # @torch.no_grad()
    # def val_test_patient(self, imgs4d, masks, times_fwd, times_bwd, ff, bf, offsets, cnn):
    #     if cnn:
    #         self.net.eval()
    #     offsets = offsets.to(torch.long)
    #     BS = offsets.shape[0]
    #     batch_indices = torch.arange(BS)

    #     m0 = masks[..., times_fwd[0].item()]
    #     mk = masks[..., times_bwd[0].item()]

    #     mts, mtts, *_ = self.time_popagation(imgs4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
    #     # loss = self.compute_loss(mts, mtts, offsets, batch_indices)
    #     # acc = self.compute_dice_acc(mts, mtts, offsets, batch_indices).item()

    #     # compute accuracy
    #     mts = mts.swapaxes(0, 1).squeeze(0)
    #     mtts = mtts.swapaxes(0, 1).squeeze(0)
    #     masks = masks[..., times_fwd[0].item(): times_fwd[-1].item() + 1]
    #     masks = masks.swapaxes(0, -1).squeeze(-1)
    #     print(masks.shape)

    #     acc_f = compute_meandice(mts, masks).mean().item()
    #     acc_b = compute_meandice(mtts, masks).mean().item()

    #     print(acc_f, acc_b)
    #     acc = 0.5 * (acc_f + acc_b)
    #     return acc

    @torch.no_grad()
    def test_patient(self, img4d, masks, times_fwd, times_bwd, ff, bf, cnn):
        if cnn:
            self.net.eval()

        BS, NZ, NY, NX, _, timesteps = ff.shape
        warp = WarpCNN(self.config, NZ, NY, NX)
        mts = torch.empty_like(masks)
        mts[..., times_fwd[0]] = masks[..., times_fwd[0]]
        mtts = torch.empty_like(masks)
        mtts[..., times_bwd[0]] = masks[..., times_bwd[0]]

        for t in range(timesteps - 1):
            # Forward mask propagation
            mt = warp(mts[..., times_fwd[t]], ff[..., t])
            if cnn:
                x = torch.cat((img4d[..., times_fwd[t + 1]], mt), dim=1)
                mts[..., times_fwd[t + 1]], _ = self.net(x)
            else:
                mts[..., times_fwd[t + 1]] = mt

            # Backward mask propagation
            mtt = warp(mtts[..., times_bwd[t]], bf[..., t])
            if cnn:
                x = torch.cat((img4d[..., times_bwd[t + 1]], mtt), dim=1)
                mtts[..., times_bwd[t + 1]], _ = self.net(x)
            else:
                mtts[..., times_bwd[t + 1]] = mtt

        # Compute forward accuracy
        mts = mts.swapaxes(0, -1).squeeze(-1)
        masks = masks.swapaxes(0, -1).squeeze(-1)
        mts = torch.where(mts > 0.5, 1.0, 0.0)
        acc_fwd = compute_meandice(mts, masks).mean().item()

        # Compute backward accuracy
        mtts = mtts.swapaxes(0, -1).squeeze(-1)
        mtts = torch.where(mtts > 0.5, 1.0, 0.0)
        acc_bwd = compute_meandice(mtts, masks).mean().item()

        # Mean acc
        acc = 0.5 * (acc_fwd + acc_bwd)
        return acc, mts, mtts
