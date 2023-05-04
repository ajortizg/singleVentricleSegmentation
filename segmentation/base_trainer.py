import os.path as osp
import sys
import time

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
import pandas as pd
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tqdm import tqdm
from terminaltables import AsciiTable
import matplotlib.pyplot as plt

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T


class BaseTrainer(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, n_classes: int, model: Module, loss_fn: _Loss, optimizer: Optimizer, device, logger=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.device = device
        self.logger = logger
        self.H = None

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

            dice = compute_dice(T.one_hot(logits, self.n_classes, argmax=True),
                                T.one_hot(label, self.n_classes),
                                include_background=False).mean()
        return loss.item(), dice.item()

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        report = pd.DataFrame(columns=['Dice', 'HD'])

        for data in loader:
            acc, hd = self._test_minibatch_impl(data)
            report.loc[len(report)] = [acc, hd]
        return report

    @abstractmethod
    def _test_minibatch_impl(self, data):
        with torch.no_grad():
            image = data['image'].to(self.device)
            label = data['label'].to(self.device)

            logits = self.model(image)

            y_pred = T.one_hot(logits, self.n_classes, argmax=True)
            y_true = T.one_hot(label, self.n_classes)
            dice = compute_dice(y_pred, y_true, include_background=False).mean()
            hd = compute_hausdorff_distance(y_pred, y_true, include_background=False).mean()
        return dice.item(), hd.item()

    def training_loop(self, epochs, scheduler, train_loader, val_loader, test_loader, tensorboard, save_dir,
                      patience, metric_early_stopping, initial_best_metric_value, comparisson_fn):
        self.H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
        epochs_since_last_improvement = 0
        best_metric = initial_best_metric_value

        tic = time.time()
        for e in tqdm(range(1, epochs + 1)):
            epoch_tic = time.time()
            report = self.train(train_loader)
            self.H['train_loss'].append(report['Loss'].mean())
            self.H['train_dice'].append(report['Dice'].mean())

            if val_loader is not None:
                report = self.validate(val_loader)
                self.H['val_loss'].append(report['Loss'].mean())
                self.H['val_dice'].append(report['Dice'].mean())
            else:
                self.H['val_loss'].append(0)
                self.H['val_dice'].append(0)

            if test_loader is not None:
                report = self.test(test_loader)
                self.H['test_dice'].append(report['Dice'].mean())
            else:
                self.H['test_dice'].append(0)

            tensorboard.add_scalar('lr', self.optimizer.param_groups[0]['lr'], e)
            if scheduler:
                scheduler.step()

            self.log(AsciiTable([
                ['Split', 'Loss', 'Dice'],
                ['Train', '{:.3f}'.format(self.H['train_loss'][-1]), '{:.3f}'.format(self.H['train_dice'][-1])],
                ['Val', '{:.3f}'.format(self.H['val_loss'][-1]), '{:.3f}'.format(self.H['val_dice'][-1])],
                ['Test', '-', '{:.3f}'.format(self.H['test_dice'][-1])],
                ['Epoch', e, epochs_since_last_improvement]
            ]).table)

            if comparisson_fn(self.H[metric_early_stopping][-1], best_metric):
                best_metric = self.H[metric_early_stopping][-1]
                epochs_since_last_improvement = 0
                self.create_checkpoint(e, osp.join(save_dir, 'checkpoint_best.pth'))
                self.log(f'Checkpoint updated with {metric_early_stopping}: {best_metric:,.3f}')
            else:
                epochs_since_last_improvement += 1

            tensorboard.add_scalars('loss', {'train': self.H['train_loss'][-1], 'val': self.H['val_loss'][-1]}, e)
            tensorboard.add_scalars('dice', {'train': self.H['train_dice'][-1], 'val': self.H['val_dice'][-1], 'test': self.H['test_dice'][-1]}, e)
            tensorboard.add_scalar('epoch_time', time.time() - epoch_tic, e)

            # early stop
            if epochs_since_last_improvement > patience:
                self.log(f'Early stop at epoch: {e}')
                break

        self.create_checkpoint(e, osp.join(save_dir, 'checkpoint_final.pth'))
        self.log('\nTraining time: {:.3f} hrs.'.format((time.time() - tic) / 3600.0))
        return self.H

    def create_checkpoint(self, e, file):
        if self.H is not None:
            torch.save({'epoch': e,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': self.H['train_loss'][-1],
                        'train_acc': self.H['train_dice'][-1],
                        'val_loss': self.H['val_loss'][-1],
                        'val_acc': self.H['val_dice'][-1],
                        'test_acc': self.H['test_dice'][-1]
                        }, file)

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def plot_loss_history(self, file):
        if self.H is not None:
            plt.figure()
            plt.plot(self.H['train_loss'], label='train')
            plt.plot(self.H['val_loss'], label='val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='lower left')
            plt.savefig(file)

    def plot_accuracy_history(self, file):
        if self.H is not None:
            plt.figure()
            plt.plot(self.H['train_dice'], label='train')
            plt.plot(self.H['val_dice'], label='val')
            plt.plot(self.H['test_dice'], label='test')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.legend(loc='lower right')
            plt.savefig(file)
