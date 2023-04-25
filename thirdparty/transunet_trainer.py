from base_trainer import BaseTrainer
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from monai.metrics.meandice import compute_dice
import torch.nn.functional as F
import os.path as osp
import sys


class TransUNetTrainer(BaseTrainer):
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device)

    def _train_minibatch_impl(self, data):
        return super()._train_minibatch_impl(data)

    def _validate_minibatch_impl(self, data):
        return super()._validate_minibatch_impl(data)

    def _test_minibatch_impl(self, data):
        return super()._test_minibatch_impl(data)
