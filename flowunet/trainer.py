import os.path as osp
import sys
import time
import os
from configparser import ConfigParser

import torch
from torch import nn
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer
from cnn.warp import WarpCNN
import segmentation.transforms as T


class FlowUNetTrainer(BaseTrainer):
    def __init__(self, config: ConfigParser, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)
        self.config = config
        self.multiclass = n_classes > 1

    def train(self, loader: DataLoader):
        self.model.train()
        report = pd.DataFrame(columns=['Loss', 'LS', 'LU', 'LP', 'Dice', 'DF', 'DB'])

        for data in loader:
            LT, LS, LU, LP, DF, DB = self._train_minibatch_impl(data)
            report.loc[len(report)] = [LT, LS, LU, LP, (DF + DB) * 0.5, DF, DB]
        return report

    def _train_minibatch_impl(self, data):
        mts, mtts, mhs, mhhs = self.propagation(data)
        offsets = torch.tensor(data['offsets']['forward_flow'], dtype=torch.long)
        batch_idxs = torch.arange(offsets.shape[0])
        LT, LS, LU, LP = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_idxs)

        self.optimizer.zero_grad()
        LT.backward()
        self.optimizer.step()

        with torch.no_grad():
            dice_fwd, dice_bwd = self.dice(mts, mtts, offsets, batch_idxs)
        return LT.item(), LS.item(), LU.item(), LP.item(), dice_fwd, dice_bwd

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        report = pd.DataFrame(columns=['Loss', 'LS', 'LU', 'LP', 'Dice', 'DF', 'DB'])

        for data in loader:
            LT, LS, LU, LP, DF, DB = self._validate_minibatch_impl(data)
            report.loc[len(report)] = [LT, LS, LU, LP, (DF + DB) * 0.5, DF, DB]
        return report

    def _validate_minibatch_impl(self, data):
        with torch.no_grad():
            mts, mtts, mhs, mhhs = self.propagation(data)
            offsets = torch.tensor(data['offsets']['forward_flow'], dtype=torch.long)
            batch_idxs = torch.arange(offsets.shape[0])
            LT, LS, LU, LP = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_idxs)
            dice_fwd, dice_bwd = self.dice(mts, mtts, offsets, batch_idxs)
        return LT.item(), LS.item(), LU.item(), LP.item(), dice_fwd, dice_bwd

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        report = pd.DataFrame(columns=['Patient', 'Dice', 'HD'])

        for data in loader:
            dice_fwd, dice_bwd, hd_fwd, hd_bwd = self._test_minibatch_impl(data)
            report.loc[len(report)] = [data['patient'][0], (dice_fwd + dice_bwd) * 0.5, (hd_fwd + hd_bwd) * 0.5]
        self.log(tabulate(report.round(3), headers='keys', tablefmt='psql'))
        return report

    def _test_minibatch_impl(self, data):
        with torch.no_grad():
            mts, mtts, *_ = self.propagation(data)
            mts = mts.squeeze(1)
            mtts = mtts.squeeze(1)
            if self.multiclass:
                mts = T.one_hot(mts, self.n_classes + 1, argmax=True)
                mtts = T.one_hot(mtts, self.n_classes + 1, argmax=True)
            else:
                mts = mts.round()
                mtts = mtts.round()

            times_fwd = data['times_fwd']
            label = data['label'][times_fwd[:, 0]].squeeze(1).to(self.device)

            dice_fwd = compute_dice(mts, label, include_background=not self.multiclass).mean().item()
            dice_bwd = compute_dice(mtts, label, include_background=not self.multiclass).mean().item()
            hd_fwd = compute_hausdorff_distance(mts, label, include_background=not self.multiclass).mean().item()
            hd_bwd = compute_hausdorff_distance(mtts, label, include_background=not self.multiclass).mean().item()
        return dice_fwd, dice_bwd, hd_fwd, hd_bwd

    def propagation(self, data):
        image = data['image'].to(self.device)
        mi, mf = data['mi'].to(self.device), data['mf'].to(self.device)
        times_fwd, times_bwd = data['times_fwd'], data['times_bwd']
        ff = data['forward_flow'].to(self.device)
        bf = data['backward_flow'].to(self.device)
        ntimes, bs, nz, ny, nx, _ = ff.shape
        batch_idxs = torch.arange(bs)

        warp = WarpCNN(self.config, nz, ny, nx)
        mts = torch.empty(size=(ntimes + 1, bs, self.n_classes if not self.multiclass else self.n_classes + 1,
                                nz, ny, nx), dtype=mi.dtype, device=self.device)
        mtts = torch.empty_like(mts)
        mhs = torch.empty(size=(ntimes, bs, self.n_classes if not self.multiclass else self.n_classes + 1,
                                nz, ny, nx), dtype=mi.dtype, device=self.device)
        mhhs = torch.empty_like(mhs)
        mts[0] = mi
        mtts[-1] = mf
        for t in range(ntimes):
            mt = warp(mts[t], ff[t])
            x = torch.cat((image[times_fwd[t + 1], batch_idxs], mt), dim=1)
            mts[t + 1], mhs[t] = self.model(x)

            mtt = warp(mtts[ntimes - t], bf[t])
            x = torch.cat((image[times_bwd[t + 1], batch_idxs], mtt), dim=1)
            mtts[ntimes - 1 - t], mhhs[ntimes - 1 - t] = self.model(x)
        return mts, mtts, mhs, mhhs

    def dice(self, mts, mtts, offsets, batch_idxs):
        mi_true = mts[0, batch_idxs]
        mi_pred = mtts[offsets[batch_idxs], batch_idxs]
        mf_true = mtts[-1, batch_idxs]
        mf_pred = mts[-offsets[batch_idxs] - 1, batch_idxs]

        if self.multiclass:
            mi_pred = T.one_hot(mi_pred, self.n_classes + 1, argmax=True)
            mf_pred = T.one_hot(mf_pred, self.n_classes + 1, argmax=True)
        else:
            mi_pred = mi_pred.round()
            mf_pred = mf_pred.round()

        dice_fwd = compute_dice(mf_pred, mf_true, include_background=not self.multiclass).mean().item()
        dice_bwd = compute_dice(mi_pred, mi_true, include_background=not self.multiclass).mean().item()
        return dice_fwd, dice_bwd
