import os.path as osp
import sys

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
import pandas as pd
from monai.metrics.meandice import compute_dice

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T


class BaseTrainer(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.device = device

    def train(self, loader: DataLoader):
        self.model.train()
        report = pd.DataFrame(columns=['Loss', 'Dice'])

        for data in loader:
            loss, acc = self._train_minibatch_impl(data)
            report.loc[len(report)] = [loss, acc]
        return report

    @abstractmethod
    def _train_minibatch_impl(self, data):
        image = data['image'].to(self.device)
        label = data['label'].to(self.device)

        logits = self.model(image)

        loss = self.loss_fn(logits, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            # probas = torch.softmax(logits[2], dim=1)
            dice = compute_dice(T.one_hot(logits, self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()

        return loss.item(), dice.item()

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        report = pd.DataFrame(columns=['Loss', 'Dice'])

        for data in loader:
            loss, acc = self._validate_minibatch_impl(data)
            report.loc[len(report)] = [loss, acc]
        return report

    @abstractmethod
    def _validate_minibatch_impl(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.model(image)
            loss = self.loss_fn(logits, label)

            # probas = torch.softmax(logits[2], dim=1)
            dice = compute_dice(T.one_hot(logits, self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()
        return loss.item(), dice.item()

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        report = pd.DataFrame(columns=['Dice'])

        for data in loader:
            acc = self._test_minibatch_impl(data)
            report.loc[len(report)] = [acc]
        return report

    @abstractmethod
    def _test_minibatch_impl(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.model(image)

            # probas = torch.softmax(logits[2], dim=1)
            dice = compute_dice(T.one_hot(logits, self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()
        return dice.item()
