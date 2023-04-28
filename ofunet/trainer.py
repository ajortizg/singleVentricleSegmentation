import os
import os.path as osp
import sys

from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from segmentation.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device, logger)
