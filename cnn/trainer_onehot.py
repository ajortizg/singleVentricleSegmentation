import os.path as osp
import sys
import torch
from torch import nn
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.transforms import RemoveSmallObjects, KeepLargestConnectedComponent
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from cnn.warp import WarpCNN
import utilities.path_utils as path_utils
from segmentation.transforms import one_hot
from cnn.loss import CustomLoss


class TrainerOneHot:
    def __init__(self, net, opt, pbar, config, device, writer, logger):
        self.net = net
        self.opt = opt
        self.pbar = pbar
        self.config = config
        self.device = device
        self.writer = writer
        self.logger = logger
        params = config['PARAMETERS']
        self.num_classes = config.getint('PARAMETERS', 'num_classes')
        self.penalization = params.getboolean('loss_penalization')

        # statistics
        self.mean_epoch_stat = {'train_loss': [(0,) * 4],
                                'train_acc': [(0,) * 3],
                                'val_loss': [(0,) * 4],
                                'val_acc': [(0,) * 3],
                                'test_acc': [(0,) * 3]}
        self.best_acc = 0.0
        self.epochs_since_last_improvement = 0

        self.loss_fn = CustomLoss(
            params.getfloat('loss_lambda'),
            params.getfloat('loss_penalization_gamma')
        )

    def last_train_accuracy(self):
        return self.mean_epoch_stat['train_acc'][-1][0]

    def last_val_accuracy(self):
        return self.mean_epoch_stat['val_acc'][-1][0]

    def last_test_accuracy(self):
        return self.mean_epoch_stat['test_acc'][-1][0]

    def last_train_loss(self):
        return self.mean_epoch_stat['train_loss'][-1][0]

    def last_val_loss(self):
        return self.mean_epoch_stat['val_loss'][-1][0]

    def train_epoch(self, train_loader):
        self.net.train()
        total_loss = (0.0, ) * 4
        total_acc = (0.0, ) * 3
        steps = len(train_loader)

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(train_loader):
            self.pbar.set_postfix_str(f'Train: {i+1}/{steps}')
            img4d = img4d.to(self.device)
            m0 = one_hot(m0, self.num_classes).to(self.device)
            mk = one_hot(mk, self.num_classes).to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)
            offsets = offsets.to(torch.long)
            batch_indices = torch.arange(offsets.shape[0])

            mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, True)
            loss = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_indices)

            self.opt.zero_grad()
            loss[0].backward()
            self.opt.step()

            with torch.no_grad():
                total_loss = tuple(tl + l.item() for tl, l in zip(total_loss, loss))
                acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
                total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc0 + acck), acc0, acck)))

        avg_loss = tuple(x / steps for x in total_loss)
        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['train_loss'].append(avg_loss)
        self.mean_epoch_stat['train_acc'].append(avg_acc)
        return (*avg_loss, avg_acc[0])

    @torch.no_grad()
    def val_epoch(self, val_loader, cnn=True):
        self.net.eval()
        total_loss = (0.0, ) * 4
        total_acc = (0.0, ) * 3
        steps = len(val_loader)

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, offsets) in enumerate(val_loader):
            self.pbar.set_postfix_str(f'Val: {i+1}/{steps}')
            img4d = img4d.to(self.device)
            m0 = one_hot(m0, self.num_classes).to(self.device)
            mk = one_hot(mk, self.num_classes).to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)
            offsets = offsets.to(torch.long)
            batch_indices = torch.arange(offsets.shape[0])

            mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn)
            loss = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_indices)

            total_loss = tuple(tl + l.item() for tl, l in zip(total_loss, loss))
            acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
            total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc0 + acck), acc0, acck)))

        avg_loss = tuple(x / steps for x in total_loss)
        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['val_loss'].append(avg_loss)
        self.mean_epoch_stat['val_acc'].append(avg_acc)
        return (*avg_loss, avg_acc[0])

    @torch.no_grad()
    def test_epoch(self, test_loader, test_ds):
        self.net.eval()
        total_acc = (0.0, ) * 3
        steps = len(test_loader)
        if steps == 0:
            self.mean_epoch_stat['test_acc'].append(total_acc)
            return total_acc

        for i, (_, img4d, _, _, masks, times_fwd, times_bwd, ff, bf, _) in enumerate(test_loader):
            self.pbar.set_postfix_str(f'Test: {i+1}/{steps}')
            times_fwd, times_bwd = test_ds.create_timeline(times_fwd[0], times_bwd[0], masks.shape[-1])
            img4d = img4d.to(self.device)
            masks = masks.squeeze(0).permute(4, 0, 1, 2, 3)
            masks = one_hot(masks, self.num_classes).to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)

            BS, NZ, NY, NX, _, timesteps = ff.shape
            warp = WarpCNN(self.config, NZ, NY, NX)
            mts = torch.empty_like(masks)
            mts[times_fwd[0]] = masks[times_fwd[0]]
            mtts = torch.empty_like(masks)
            mtts[times_bwd[0]] = masks[times_bwd[0]]

            for t in range(timesteps - 1):
                # Forward mask propagation
                mt = warp(mts[times_fwd[t]].unsqueeze(0), ff[..., t])
                x = torch.cat((img4d[..., times_fwd[t + 1]], mt), dim=1)
                mts[times_fwd[t + 1]], _ = self.net(x)

                # Backward mask propagation
                mtt = warp(mtts[times_bwd[t]].unsqueeze(0), bf[..., t])
                x = torch.cat((img4d[..., times_bwd[t + 1]], mtt), dim=1)
                mtts[times_bwd[t + 1]], _ = self.net(x)

            ti, tf = times_fwd[0], times_bwd[0]
            mts = mts[ti:tf + 1]
            mtts = mtts[ti:tf + 1]
            masks = masks[ti:tf + 1]

            # Compute forward accuracy
            mts = one_hot(mts, self.num_classes, argmax=True)
            acc_fwd = compute_dice(mts, masks, include_background=False).mean().item()

            # Compute backward accuracy
            mtts = one_hot(mtts, self.num_classes, argmax=True)
            acc_bwd = compute_dice(mtts, masks, include_background=False).mean().item()

            total_acc = tuple(ta + a for ta, a in zip(total_acc, (0.5 * (acc_fwd + acc_bwd), acc_bwd, acc_fwd)))

        avg_acc = tuple(x / steps for x in total_acc)
        self.mean_epoch_stat['test_acc'].append(avg_acc)
        return total_acc

    @torch.no_grad()
    def cardiac_cycle_propagation(self, dset, loader):
        self.net.eval()
        steps = len(loader)
        if steps == 0:
            return None

        for i, (_, img4d, m0, mk, _, times_fwd, times_bwd, ff, bf, _) in enumerate(loader):
            self.pbar.set_postfix_str(f'ccc: {i+1}/{steps}')
            img4d = img4d.to(self.device)
            m0 = m0.to(self.device)
            mk = mk.to(self.device)
            ff = ff.to(self.device)
            bf = bf.to(self.device)

            BS, NZ, NY, NX, _, timesteps = ff.shape
            times_fwd, times_bwd = dset.create_timeline(times_fwd[0], times_bwd[0], timesteps)
            warp = WarpCNN(self.config, NZ, NY, NX)
            mts = torch.empty(size=(BS, 1, NZ, NY, NX, timesteps), device=self.device, dtype=m0.dtype)
            mts[..., times_fwd[0]] = m0
            mtts = torch.empty_like(mts)
            mtts[..., times_bwd[0]] = mk

            for t in range(timesteps - 1):
                # Forward mask propagation
                mt = warp(mts[..., times_fwd[t]], ff[..., t])
                x = torch.cat((img4d[..., times_fwd[t + 1]], mt), dim=1)
                mts[..., times_fwd[t + 1]], _ = self.net(x)

                # Backward mask propagation
                mtt = warp(mtts[..., times_bwd[t]], bf[..., t])
                x = torch.cat((img4d[..., times_bwd[t + 1]], mtt), dim=1)
                mtts[..., times_bwd[t + 1]], _ = self.net(x)

        return mts, mtts

    def normalize(self, x):
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(3):
            x[..., i] = 2 * (x[..., i] / (80 - 1) - 0.5)
        return x

    def time_popagation(self, img4d, m0, mk, times_fwd, times_bwd, ff, bf, batch_indices, cnn=True):
        BS, NZ, NY, NX, CH, timesteps = ff.shape
        dtype = m0.dtype
        warp = WarpCNN(self.config, NZ, NY, NX)
        mts = torch.empty(size=(timesteps + 1, BS, self.num_classes, NZ, NY, NX), dtype=dtype, device=self.device)
        mts[0] = m0
        mtts = torch.empty_like(mts)
        mtts[-1] = mk

        if self.penalization:
            mhs = torch.empty(size=(timesteps, BS, self.num_classes, NZ, NY, NX), dtype=dtype, device=self.device)
            mhhs = torch.empty_like(mhs)
        else:
            mhs = mhhs = None

        for t in range(timesteps):
            # Forward propagation m0 -> mk
            mt = warp(mts[t], ff[..., t])
            if cnn:
                x = torch.cat((img4d[batch_indices, ..., times_fwd[t + 1][batch_indices]], mt), dim=1)
                mts[t + 1], mh = self.net(x)
            else:
                mts[t + 1] = mt

            # Backward propagation mk -> m0
            mtt = warp(mtts[timesteps - t], bf[..., t])
            if cnn:
                x = torch.cat((img4d[batch_indices, ..., times_bwd[t + 1][batch_indices]], mtt), dim=1)
                mtts[timesteps - t - 1], mhh = self.net(x)
            else:
                mtts[timesteps - t - 1] = mtt

            if self.penalization and cnn:
                mhs[t] = mh
                mhhs[t] = mhh
        return (mts, mtts, mhs, mhhs)

    def log(self, e):
        self.logger.info('Epoch: %d' % e)
        for k, v in self.mean_epoch_stat.items():
            if len(v) > 0:
                self.logger.info(f'\t{k}:\t{self.__tuple_float_to_str(v[-1])}')

        # Plot accuracy in tensorboard
        for i, (train, val, test) in enumerate(
            zip(self.mean_epoch_stat['train_acc'][-1],
                self.mean_epoch_stat['val_acc'][-1],
                self.mean_epoch_stat['test_acc'][-1])):
            self.writer.add_scalars(f'acc/a{i}', {'train': train, 'val': val, 'test': test}, e)

        # Plot loss in tensorboard
        for i, (train, val) in enumerate(
            zip(self.mean_epoch_stat['train_loss'][-1],
                self.mean_epoch_stat['val_loss'][-1])):
            self.writer.add_scalars(f'loss/l{i}', {'train': train, 'val': val}, e)

        # Plot learning rate in tensorboard
        self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], e)

    def __tuple_float_to_str(self, t, precision=3, sep=', '):
        return '{}'.format(sep.join(f'{x:.{precision}f}' for x in t))

    def create_checkpoint(self, e, save_dir, when_better, which='val', verbose=False):
        if when_better:
            last_acc = self.last_test_accuracy() if which == 'test' else self.last_val_accuracy()
            if last_acc > self.best_acc:
                self.best_acc = last_acc
                self.epochs_since_last_improvement = 0
                self.checkpoint(e, save_dir, f'checkpoint_best.pth')
                self.save_stats(save_dir)
                if verbose:
                    self.logger.info(f'\t{which} chkpt updated with acc: {self.best_acc:,.3f}')
            else:
                self.epochs_since_last_improvement += 1
        else:
            self.checkpoint(e, save_dir, 'checkpoint_final.pth')
            if verbose:
                self.logger.info(f'\tSaved chkpt')

    def checkpoint(self, e, save_dir, filename):
        torch.save({
            'epoch': e,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'train_loss': self.last_train_loss(),
            'train_acc': self.last_train_accuracy(),
            'val_loss': self.last_val_loss(),
            'val_acc': self.last_val_accuracy(),
            'test_acc': self.last_test_accuracy()
        }, osp.join(save_dir, filename))

    def save_stats(self, save_dir):
        for k, v in self.mean_epoch_stat.items():
            path_utils.write_list(save_dir, k + '.csv', v)

    def compute_dice_acc(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        # m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)
        m0tt = one_hot(m0tt, self.num_classes, argmax=True)

        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = one_hot(mkt, self.num_classes, argmax=True)
        # mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        acc0 = compute_dice(m0tt, m0, include_background=False).mean()
        acck = compute_dice(mkt, mk, include_background=False).mean()
        # dc = 0.5 * (acc0 + acck)
        return acc0.item(), acck.item()

    def compute_hd(self, mts, mtts, offsets, batch_indices):
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        m0tt = torch.where(m0tt > 0.5, 1.0, 0.0)

        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        mkt = torch.where(mkt > 0.5, 1.0, 0.0)

        hd0 = compute_hausdorff_distance(m0tt, m0).mean().item()
        hdk = compute_hausdorff_distance(mkt, mk).mean().item()
        return hd0, hdk

    def train_patient(self, img4d, m0, mk, timesfwd, timesbwd, ff, bf):
        BS = 1
        offsets = torch.zeros(BS, dtype=torch.long)
        batch_indices = torch.arange(BS)

        self.net.train()
        mts, mtts, mhs, mhhs = self.time_popagation(img4d, m0, mk, timesfwd, timesbwd, ff, bf, batch_indices)
        loss = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_indices)

        self.opt.zero_grad()
        loss[0].backward()
        self.opt.step()

        with torch.no_grad():
            acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)

        avg_loss = tuple(x.item() for x in loss)
        avg_acc = (0.5 * (acc0 + acck), acc0, acck)
        self.mean_epoch_stat['train_loss'].append(avg_loss)
        self.mean_epoch_stat['train_acc'].append(avg_acc)
        return (*avg_loss, avg_acc[0])

    @torch.no_grad()
    def val_patient(self, imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, cnn):
        BS = 1
        offsets = torch.zeros(BS, dtype=torch.long)
        batch_indices = torch.arange(BS)

        self.net.eval()
        mts, mtts, *_ = self.time_popagation(imgs4d, m0s, mks, times_fwd, times_bwd, ff, bf, batch_indices, cnn)

        acc0, acck = self.compute_dice_acc(mts, mtts, offsets, batch_indices)
        mean_acc = 0.5 * (acc0 + acck)

        hd0, hdk = self.compute_hd(mts, mtts, offsets, batch_indices)
        mean_hd = 0.5 * (hd0 + hdk)
        metrics = {'mean_acc': mean_acc, 'acc_fwd': acck, 'acc_bwd': acc0,
                   'mean_hd': mean_hd, 'hd_fwd': hdk, 'hd_bwd': hd0}
        return metrics, mts, mtts

    @torch.no_grad()
    def test_patient(self, img4d, masks, times_fwd, times_bwd, ff, bf, cnn, hd=True):
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

        ti = times_fwd[0]
        tf = times_bwd[0]

        # Compute forward accuracy
        mts = mts.swapaxes(0, -1).squeeze(-1)
        masks = masks.swapaxes(0, -1).squeeze(-1)
        mts = torch.where(mts > 0.5, 1.0, 0.0)
        mts_es_ed = mts[ti:tf + 1]
        mask_es_ed = masks[ti:tf + 1]
        accs_fwd = compute_dice(mts_es_ed, mask_es_ed)
        mean_acc_fwd = accs_fwd.mean().item()

        # Compute backward accuracy
        mtts = mtts.swapaxes(0, -1).squeeze(-1)
        mtts = torch.where(mtts > 0.5, 1.0, 0.0)
        mtts_es_ed = mtts[ti:tf + 1]
        accs_bwd = compute_dice(mtts_es_ed, mask_es_ed)
        mean_acc_bwd = accs_bwd.mean().item()

        if hd:
            hd_fwd = compute_hausdorff_distance(mts_es_ed, mask_es_ed)
            mean_hd_fwd = hd_fwd.mean().item()
            hd_bwd = compute_hausdorff_distance(mtts_es_ed, mask_es_ed)
            mean_hd_bwd = hd_bwd.mean().item()
        else:
            hd_fwd = torch.tensor([0.0], device=self.device)
            mean_hd_fwd = 0
            hd_bwd = torch.tensor([0.0], device=self.device)
            mean_hd_bwd = 0

        metrics = {'mean_acc': 0.5 * (mean_acc_fwd + mean_acc_bwd),
                   'mean_acc_fwd': mean_acc_fwd,
                   'mean_acc_bwd': mean_acc_bwd,
                   'mean_hd': 0.5 * (mean_hd_fwd + mean_hd_bwd)}
        self.mean_epoch_stat['test_acc'].append((metrics['mean_acc'], metrics['mean_acc_fwd'], metrics['mean_acc_bwd']))
        return (metrics,
                accs_fwd.squeeze().detach().cpu().tolist(),
                accs_bwd.squeeze().detach().cpu().tolist(),
                mts,
                mtts,
                hd_fwd.squeeze().detach().cpu().tolist(),
                hd_bwd.squeeze().detach().cpu().tolist())

    def plot_stats(self, save_dir, log_scale=False):
        # Plot accuracy
        # fast
        plt.style.use('seaborn-paper')
        plt.figure()
        plt.plot(np.array(self.mean_epoch_stat['train_acc'])[1:, 0], label='Train')
        plt.plot(np.array(self.mean_epoch_stat['val_acc'])[1:, 0], label='Val')
        plt.plot(np.array(self.mean_epoch_stat['test_acc'])[1:, 0], label='Test')
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
        plt.plot(np.array(self.mean_epoch_stat['train_loss'])[1:, 0], label='Train')
        plt.plot(np.array(self.mean_epoch_stat['val_loss'])[1:, 0], label='Val')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))
