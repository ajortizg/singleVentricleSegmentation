import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer


class TransUNetTrainer(BaseTrainer):
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)

    def _train_minibatch_impl(self, data):
        return super()._train_minibatch_impl(data)

    def _validate_minibatch_impl(self, data):
        return super()._validate_minibatch_impl(data)

    def _test_minibatch_impl(self, data):
        return super()._test_minibatch_impl(data)
