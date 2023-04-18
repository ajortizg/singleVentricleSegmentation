import os.path as osp
import sys
import torch
from torch import nn
from monai.metrics.meandice import compute_meandice
import numpy as np
import os
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.warp import WarpCNN
from utilities import plots


__all__ = ['Trainer']


class Trainer:
    def __init__(self, net, opt, pbar, config, device, writer, logger, display_prob=0.2):
        self.net = net
        self.opt = opt
        self.pbar = pbar
        self.config = config
        self.device = device
        self.writer = writer
        self.logger = logger
        self.display_prob = display_prob
        self.loss_lambda = config.getfloat('PARAMETERS', 'LOSS_LAMBDA')
        loss_fn_type = config.get('PARAMETERS', 'LOSS_FN')
        self.penalization = config.getboolean('PARAMETERS', 'LOSS_PENALIZATION')
        self.mu = config.getfloat('PARAMETERS', 'LOSS_PENALIZATION_MU')
        self.reduction = config.get('PARAMETERS', 'LOSS_REDUCTION')
        self.loss_fn = None

        # statistics
        self.mean_epoch_stat = {'train_loss': [],
                                'train_acc': [],
                                'val_loss': [],
                                'val_acc': [],
                                'test_acc': []}
        self.best_val_acc = 0.0

        if loss_fn_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=self.reduction)
        elif loss_fn_type == 'huber':
            huber_delta = config.getfloat('PARAMETERS', 'HUBER_DELTA')
            self.loss_fn = nn.HuberLoss(reduction=self.reduction, delta=huber_delta)
        else:
            print('Unknown loss function: ' + loss_fn_type)
            sys.exit()

    def train_epoch(self, train_loader, cnn=True):
        self.net.train()
        total_loss = (0.0, 0.0, 0.0, 0.0, 0.0)
        total_acc = (0.0, 0.0, 0.0)
        steps = len(train_loader)

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(train_loader):
            self.pbar.set_postfix_str(f'Train: {i+1}/{steps}')
            img4d = img4d.to(self.device)
            m0 = m0.to(self.device)
            mk = mk.to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)
            offsets = offsets.to(torch.long)
            batch_indices = torch.arange(offsets.shape[0])

            mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
            loss = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

            self.opt.zero_grad()
            loss[0].backward()
            self.opt.step()

            with torch.no_grad():
                total_loss = tuple(tl + l.item() for tl, l in zip(total_loss, loss))
                acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
                total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc0 + acck), acc0, acck)))
                if np.random.rand() < self.display_prob:
                    self.plot_imgs(mts, mtts, offsets, batch_indices, 'train')

        avg_loss = tuple(x / steps for x in total_loss)
        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['train_loss'].append(avg_loss)
        self.mean_epoch_stat['train_acc'].append(avg_acc)
        return (*avg_loss, avg_acc[0])

    @torch.no_grad()
    def val_epoch(self, val_loader, cnn=True):
        self.net.eval()
        total_loss = (0.0, 0.0, 0.0, 0.0, 0.0)
        total_acc = (0.0, 0.0, 0.0)
        steps = len(val_loader)

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(val_loader):
            self.pbar.set_postfix_str(f'Val: {i+1}/{steps}')
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
            acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
            total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc0 + acck), acc0, acck)))
            if np.random.rand() < self.display_prob:
                self.plot_imgs(mts, mtts, offsets, batch_indices, 'val')

        avg_loss = tuple(x / steps for x in total_loss)
        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['val_loss'].append(avg_loss)
        self.mean_epoch_stat['val_acc'].append(avg_acc)
        return (*avg_loss, avg_acc[0])

    @torch.no_grad()
    def test_epoch(self, test_loader, test_ds):
        self.net.eval()
        total_acc = (0.0, 0.0, 0.0)
        steps = len(test_loader)
        if steps == 0:
            self.mean_epoch_stat['test_acc'].append(total_acc)
            return total_acc

        for i, (_, img4d, _, _, masks, times_fwd, times_bwd, ff, bf, _) in enumerate(test_loader):
            self.pbar.set_postfix_str(f'Test: {i+1}/{steps}')
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

            total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc_fwd + acc_bwd), acc_bwd, acc_fwd)))

            if np.random.rand() < self.display_prob:
                self.plot_test_imgs(mts, masks, 'test')

        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['test_acc'].append(avg_acc)
        return total_acc

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

    def log(self, e):
        self.logger.info('Epoch: %d' % e)
        for k, v in self.mean_epoch_stat.items():
            if len(v) > 0:
                self.logger.info(f'\t{k}:\t{self.tuple_float_to_str(v[-1])}')

        # Plot accuracy in tensorboard
        for i, (train, val, test) in enumerate(
            zip(self.mean_epoch_stat['train_acc'][-1],
                self.mean_epoch_stat['val_acc'][-1],
                self.mean_epoch_stat['test_acc'][-1])):
            sub = 't' if i == 0 else f'{i}'
            self.writer.add_scalars(f'acc/a{sub}', {'train': train, 'val': val, 'test': test}, e)

        # Plot loss in tensorboard
        for i, (train, val) in enumerate(zip(self.mean_epoch_stat['train_loss'][-1], self.mean_epoch_stat['val_loss'][-1])):
            sub = 't' if i == 0 else f'{i}'
            self.writer.add_scalars(f'loss/l{sub}', {'train': train, 'val': val}, e)

        # Plot learning rate in tensorboard
        self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], e)

    def tuple_float_to_str(self, t, precision=3, sep=', '):
        return '{}'.format(sep.join(f'{x:.{precision}f}' for x in t))

    def create_checkpoint(self, e, save_dir, when_better, verbose=False):
        if when_better:
            last_val_acc = self.mean_epoch_stat['val_acc'][-1][0]
            if last_val_acc > self.best_val_acc:
                self.best_val_acc = last_val_acc
                self.checkpoint(e, save_dir, 'best_val_checkpoint.pth')
                if verbose:
                    self.logger.info(f'\tVal chkpt updated with acc: {self.best_val_acc:,.3f}')
        else:
            self.checkpoint(e, save_dir, 'checkpoint.pth')
            if verbose:
                self.logger.info(f'\tSaved chkpt')

    def checkpoint(self, e, save_dir, filename):
        torch.save({
            'epoch': e,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'train_loss': self.mean_epoch_stat['train_loss'][-1][0],
            'train_acc': self.mean_epoch_stat['train_acc'][-1][0],
            'val_loss': self.mean_epoch_stat['val_loss'][-1][0],
            'val_acc': self.mean_epoch_stat['val_acc'][-1][0],
            'test_acc': self.mean_epoch_stat['test_acc'][-1][0]
        }, osp.join(save_dir, filename))

    def save_stats(self, save_dir):
        for k, v in self.mean_epoch_stat.items():
            plots.write_list(save_dir, k + '.csv', v)

    def compute_dice_acc(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        acc0 = compute_meandice(m0tt, m0).mean()
        acck = compute_meandice(mkt, mk).mean()
        # dc = 0.5 * (acc0 + acck)
        return acc0.item(), acck.item()

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

    def train_patient(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, offsets):
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)

        self.net.train()
        mts, mtts, mhs, mhhs = self.time_popagation(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices)
        loss = self.compute_loss(mts, mtts, offsets, batch_indices, mhs, mhhs)

        self.opt.zero_grad()
        loss[0].backward()
        self.opt.step()

        with torch.no_grad():
            acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
            mean_acc = 0.5 * (acc0 + acck)
            loss = tuple(l.item() for l in loss)
            if np.random.rand() < self.display_prob:
                self.plot_imgs(mts, mtts, offsets, batch_indices, 'train')
        return (*loss, mean_acc)

    @torch.no_grad()
    def val_patient(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, offsets, cnn):
        self.net.eval()
        offsets = offsets.to(torch.long)
        BS = offsets.shape[0]
        batch_indices = torch.arange(BS)

        mts, mtts, *_ = self.time_popagation(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices, cnn)

        acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
        mean_acc = 0.5 * (acc0 + acck)
        return mean_acc, mts, mtts

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

    def plot_stats(self, save_dir, log_scale=False):
        # Plot accuracy
        # fast
        plt.style.use('seaborn-paper')
        plt.figure()
        plt.plot(np.array(self.mean_epoch_stat['train_acc'])[:, 0], label='Train')
        plt.plot(np.array(self.mean_epoch_stat['val_acc'])[:, 0], label='Val')
        plt.plot(np.array(self.mean_epoch_stat['test_acc'])[:, 0], label='Test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Dice')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir, 'acc.pdf'))

        # Plot loss
        plt.style.use('seaborn-paper')
        plt.figure()
        if log_scale:
            plt.yscale('log')
        plt.plot(np.array(self.mean_epoch_stat['train_loss'])[:, 0], label='Train')
        plt.plot(np.array(self.mean_epoch_stat['val_loss'])[:, 0], label='Val')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))

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
