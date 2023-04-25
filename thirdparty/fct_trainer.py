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

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T


class FCTTrainer(BaseTrainer):
    def __init__(self, img_size: int, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device=None):
        super().__init__(n_classes, model, loss_fn, optimizer, device)
        self.img_size = img_size

    def _train_minibatch_impl(self, data):
        image = data['image'].to(self.device)
        label = data['label'].to(self.device)

        logits = self.model(image)

        down1 = F.interpolate(label, self.img_size // 2, mode='nearest')
        down2 = F.interpolate(label, self.img_size // 4, mode='nearest')
        loss = (self.loss_fn(logits[2], label) * 0.57
                + self.loss_fn(logits[1], down1) * 0.29
                + self.loss_fn(logits[0], down2) * 0.14)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            # probas = torch.softmax(logits[2], dim=1)
            dice = compute_dice(T.one_hot(logits[2], self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()

        return loss.item(), dice.item()

    def _validate_minibatch_impl(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.model(image)

            down1 = F.interpolate(label, self.img_size // 2, mode='nearest')
            down2 = F.interpolate(label, self.img_size // 4, mode='nearest')
            loss = (self.loss_fn(logits[2], label) * 0.57
                    + self.loss_fn(logits[1], down1) * 0.29
                    + self.loss_fn(logits[0], down2) * 0.14)

            dice = compute_dice(T.one_hot(logits[2], self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()
        return loss.item(), dice.item()

    def _test_minibatch_impl(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.model(image)

            # probas = torch.softmax(logits[2], dim=1)
            dice = compute_dice(T.one_hot(logits[2], self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()
        return dice.item()
