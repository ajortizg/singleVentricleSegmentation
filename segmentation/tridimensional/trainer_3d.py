import sys
import os
import os.path as osp
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import torch
import pandas as pd
from tabulate import tabulate

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer
from datasets.flowunet_dataset import get_bounds


class Trainer3d(BaseTrainer):
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)

    def _train_minibatch_impl(self, data):
        return super()._train_minibatch_impl(data)

    def _validate_minibatch_impl(self, data):
        return super()._validate_minibatch_impl(data)

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        report = pd.DataFrame(columns=['Patient', 'Dice', 'HD'])

        for data in loader:
            *_, indices = get_bounds(data['es'].item(), data['ed'].item(), None, fwd=True)
            data['image'] = data['image'].squeeze(0)[indices]
            data['label'] = data['label'].squeeze(0)[indices]
            acc, hd = self._test_minibatch_impl(data)
            report.loc[len(report)] = [data['patient'][0], acc, hd]
        self.log(tabulate(report.round(3), headers='keys', tablefmt='psql'))
        return report

    def _test_minibatch_impl(self, data):
        return super()._test_minibatch_impl(data)


class FineTuner3d(Trainer3d):
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)
    
    def train(self, loader):
        self.model.train()
        report = pd.DataFrame(columns=['Loss', 'Dice'])

        for data in loader:
            t = [data['es'].item(), data['ed'].item()]
            data['image'] = data['image'].squeeze(0)[t]
            data['label'] = data['label'].squeeze(0)[t]
            loss, acc = self._train_minibatch_impl(data)
            report.loc[len(report)] = [loss, acc]
        return report

