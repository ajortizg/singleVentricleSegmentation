import os
import os.path as osp
import sys
import time
from configparser import ConfigParser

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import pandas as pd
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tabulate import tabulate

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer
import segmentation.transforms as T
from datasets.flow_unet_dataset import get_bounds
from cnn.warp import WarpCNN


class Trainer(BaseTrainer):
    def __init__(self, cfg: ConfigParser, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        self.cfg = cfg
        params = cfg['PARAMETERS']
        self.loss_lambda = params.getfloat('loss_lambda')
        self.penalization = params.getboolean('loss_penalization')
        self.loss_gamma = params.getfloat('loss_penalization_gamma')
        super().__init__(params.getint('num_classes'), model, loss_fn, optimizer, device, logger)

    def train(self, loader: DataLoader):
        self.model.train()
        report = pd.DataFrame(columns=['Loss', 'LS', 'LU', 'LP', 'Dice', 'DF', 'DB'])

        for data in loader:
            LT, LS, LU, LP, DF, DB = self._train_minibatch_impl(data)
            report.loc[len(report)] = [LT, LS, LU, LP, (DF + DB) * 0.5, DF, DB]
        return report

    def _train_minibatch_impl(self, data):
        forward_masks, backward_masks, forward_hats, backward_hats = self.propagation(data)
        LT, LS, LU, LP = self.compute_loss(forward_masks, backward_masks, forward_hats, backward_hats)

        self.optimizer.zero_grad()
        LT.backward()
        self.optimizer.step()

        with torch.no_grad():
            dice_fwd, dice_bwd = self.dice(forward_masks, backward_masks)
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
            forward_masks, backward_masks, forward_hats, backward_hats = self.propagation(data)
            LT, LS, LU, LP = self.compute_loss(forward_masks, backward_masks, forward_hats, backward_hats)
            dice_fwd, dice_bwd = self.dice(forward_masks, backward_masks)
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
            forward_masks, backward_masks, *_ = self.propagation(data, test=True)

            *_, indices = get_bounds(data['es'].item(), data['ed'].item(), None, fwd=True, test=True)
            label = data['label'].squeeze(0)[indices].to(self.device)

            pred_fwd = T.one_hot(forward_masks, self.n_classes, argmax=True)
            pred_bwd = T.one_hot(backward_masks, self.n_classes, argmax=True)
            dice_fwd = compute_dice(pred_fwd, label, include_background=False).mean().item()
            dice_bwd = compute_dice(pred_bwd, label, include_background=False).mean().item()
            hd_fwd = compute_hausdorff_distance(pred_fwd, label, include_background=False).mean().item()
            hd_bwd = compute_hausdorff_distance(pred_bwd, label, include_background=False).mean().item()
        return dice_fwd, dice_bwd, hd_fwd, hd_bwd

    def propagation(self, data, test=False):
        image = data['image'].squeeze(0)
        label = data['label'].squeeze(0)
        ff = data['forward_flow'].squeeze(0).to(self.device)
        bf = data['backward_flow'].squeeze(0).to(self.device)

        *_, mi, mf, indices = get_bounds(data['es'].item(), data['ed'].item(), label, fwd=True, test=test)
        mi = mi.unsqueeze(0).to(self.device)
        mf = mf.unsqueeze(0).to(self.device)
        image = image[indices].to(self.device)

        nz, ny, nx = mi.shape[2:]
        warp = WarpCNN(self.cfg, nz, ny, nx)
        forward_masks = [mi]    # mi_true, mi+1, mi+2, ..., mf-1, mf_est
        backward_masks = [mf]   # mf_true, mf-1, mf-2, ..., mi+1, mi_est
        forward_hats = []
        backward_hats = []

        for t in range(0, len(indices) - 1):
            mt = warp(forward_masks[-1], ff[t].unsqueeze(0))
            x = torch.cat((image[t + 1].unsqueeze(0), mt), dim=1)
            mt, mh = self.model(x)
            forward_masks.append(mt)

            mtt = warp(backward_masks[-1], bf[t].unsqueeze(0))
            x = torch.cat((image[t - 2].unsqueeze(0), mtt), dim=1)
            mtt, mhh = self.model(x)
            backward_masks.append(mtt)

            if self.penalization:
                forward_hats.append(mh)
                backward_hats.append(mhh)

        assert len(forward_masks) == len(backward_masks)
        backward_masks.reverse()    # mi_est, mi+1, mi+2, ..., mf-1, mf_true
        backward_hats.reverse()
        return torch.cat(forward_masks), torch.cat(backward_masks), torch.cat(forward_hats), torch.cat(backward_hats)

    def compute_loss(self, forward_masks, backward_masks, forward_hats, backward_hats):
        # Supervised term
        mi_true = forward_masks[0].unsqueeze(0)
        mi_pred = backward_masks[0].unsqueeze(0)
        mf_true = backward_masks[-1].unsqueeze(0)
        mf_pred = forward_masks[-1].unsqueeze(0)
        LS = self.loss_fn(mi_pred, mi_true) + self.loss_fn(mf_pred, mf_true)

        # Self-supervised term
        forward_masks = forward_masks[1:-1]
        backward_masks = backward_masks[1:-1]
        k = forward_masks.shape[0]
        LU = self.loss_fn(forward_masks, backward_masks) * (self.loss_lambda / k)

        # Penalization term
        if self.penalization:
            LP = (forward_hats.pow(2).sum() + backward_hats.pow(2).sum()) * (self.loss_gamma / k)
        else:
            LP = torch.tensor([0.0], dtype=LS.dtype, device=LS.device)

        # Compute total loss
        LT = LS + LU + LP
        return LT, LS, LU, LP

    def dice(self, forward_masks, backward_masks):
        mi_true = forward_masks[0].unsqueeze(0)
        mi_pred = backward_masks[0].unsqueeze(0)
        mf_true = backward_masks[-1].unsqueeze(0)
        mf_pred = forward_masks[-1].unsqueeze(0)

        dice_fwd = compute_dice(T.one_hot(mf_pred, self.n_classes, argmax=True),
                                mf_true,
                                include_background=False).mean().item()
        dice_bwd = compute_dice(T.one_hot(mi_pred, self.n_classes, argmax=True),
                                mi_true,
                                include_background=False).mean().item()
        return dice_fwd, dice_bwd