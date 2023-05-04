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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer
from cnn.warp import WarpCNN
import segmentation.transforms as T


class Trainer(BaseTrainer):
    def __init__(self, config: ConfigParser, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)
        self.config = config

    def train(self, loader: DataLoader):
        self.model.train()
        # report = pd.DataFrame(columns=['Loss', 'LS', 'LU', 'LP', 'Dice', 'DF', 'DB'])

        for data in loader:
            LT, LS, LU, LP, DF, DB = self._train_minibatch_impl(data)
            # report.loc[len(report)] = [LT, LS, LU, LP, (DF + DB) * 0.5, DF, DB]
        # return report

    def _train_minibatch_impl(self, data):
        mts, mtts, mhs, mhhs = self.propagation(data)
        offsets = torch.tensor(data['offsets']['forward_flow'], dtype=torch.long)
        batch_idxs = torch.arange(offsets.shape[0])
        LT, LS, LU, LP = self.loss_fn(mts, mtts, mhs, mhhs, offsets, batch_idxs)

        self.optimizer.zero_grad()
        LT.backward() 
        self.optimizer.step()

        # with torch.no_grad():
        #     dice_fwd, dice_bwd = self.dice(forward_masks, backward_masks)
        # return LT.item(), LS.item(), LU.item(), LP.item(), dice_fwd, dice_bwd

    def propagation(self, data):
        image = data['image'].to(self.device)
        mi, mf = data['mi'], data['mf']
        times_fwd, times_bwd = data['times_fwd'], data['times_bwd']
        ff = data['forward_flow'].to(self.device)
        bf = data['backward_flow'].to(self.device)
        # offsets = torch.tensor(data['offsets']['forward_flow'], dtype=torch.long)
        ntimes, bs, nz, ny, nx, _ = ff.shape
        batch_idxs = torch.arange(bs)

        mi = T.one_hot(mi[0], self.n_classes).to(self.device)
        mf = T.one_hot(mf[0], self.n_classes).to(self.device)

        warp = WarpCNN(self.config, nz, ny, nx)
        mts = torch.empty(size=(ntimes + 1, bs, self.n_classes, nz, ny, nx), dtype=mi.dtype, device=self.device)
        mtts = torch.empty_like(mts)
        mhs = torch.empty(size=(ntimes, bs, self.n_classes, nz, ny, nx), dtype=mi.dtype, device=self.device)
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

    def _validate_minibatch_impl(self, data):
        return super()._validate_minibatch_impl(data)

    def _test_minibatch_impl(self, data):
        return super()._test_minibatch_impl(data)
