import os.path as osp
import configparser
import time
import sys

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import pandas as pd
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.file_paths_utils as fpu
from utilities import stuff
from cnn.models.model_factory import create_model, save_model
from segmentation.dataset import SegmentationDataset
import segmentation.utils as utils


def create_dataloaders(config, fold):
    train_transforms, val_transforms, test_transforms = utils.get_transforms(config)

    root_dir = config.get('DATA', 'root_dir')
    batch_size = config.getint('PARAMETERS', 'batch_size')
    workers = config.getint('PARAMETERS', 'num_workers')

    train_ds = SegmentationDataset(root_dir, 'train', train_transforms, fold['train'])
    val_ds = SegmentationDataset(root_dir, 'train', val_transforms, fold['val'])
    test_ds = SegmentationDataset(root_dir, 'test', test_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train(net, loss_fn, opt, loader, device):
    net.train()
    report = pd.DataFrame(columns=['Loss', 'Dice'])

    for data in loader:
        img = data['image'].to(device)
        label = data['label'].to(device)

        logits, _ = net(img)
        loss = loss_fn(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = torch.softmax(logits, dim=1)
            # # Get the indices for the max prob along chan dim
            # indices = torch.argmax(pred, dim=1, keepdim=True)
            # # The max prob is set to 1.0
            # pred.scatter_(1, indices, 1.0)
            # pred = torch.where(pred == 1.0, 1.0, 0.0)
            pred = torch.where(pred > 0.5, 1.0, 0.0)
            dice = compute_dice(pred, label, include_background=False).mean()
            report.loc[len(report)] = [loss.item(), dice.item()]
    return report


@torch.no_grad()
def validate(net, loss_fn, loader, device):
    net.eval()
    report = pd.DataFrame(columns=['Loss', 'Dice'])

    for data in loader:
        img = data['image'].to(device)
        label = data['label'].to(device)

        logits, _ = net(img)
        loss = loss_fn(logits, label)

        pred = torch.softmax(logits, dim=1)
        # indices = torch.argmax(pred, dim=1, keepdim=True)
        # pred.scatter_(1, indices, 1.0)
        # pred = torch.where(pred == 1.0, 1.0, 0.0)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        dice = compute_dice(pred, label, include_background=False).mean()
        report.loc[len(report)] = [loss.item(), dice.item()]
    return report


@torch.no_grad()
def test(net, loader, device):
    net.eval()
    report = pd.DataFrame(columns=['Dice'])

    for data in loader:
        img = data['image'].to(device).squeeze(0)
        label = data['label'].to(device).squeeze(0)

        logits, _ = net(img)
        pred = torch.softmax(logits, dim=1)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        # indices = torch.argmax(pred, dim=1, keepdim=True)
        # pred.scatter_(1, indices, 1.0)
        # pred = torch.where(pred == 1.0, 1.0, 0.0)
        dice = compute_dice(pred, label, include_background=False).mean()
        report.loc[len(report)] = [dice.item()]
    return report


if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/seg_train.ini')
    data_cfg = config['DATA']

    fold, n = utils.get_fold(data_cfg)
    train_loader, val_loader, test_loader = create_dataloaders(config, fold)

    save_dir = fpu.create_save_dir(data_cfg['output_dir'], f'fold_{n}')
    fpu.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    fpu.save_transforms_to_json(train_loader.dataset.transforms, osp.join(save_dir, 'train_transforms.json'))
    fpu.save_json([fold], osp.join(save_dir, 'fold.json'), default=int)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))
    logger.info('Fold: {}'.format(n))

    net = create_model(config, logger).to(device)
    save_model(net, save_dir, 'net.txt')
    loss_fn = utils.get_loss_fn(config)
    optimizer = optim.Adam(net.parameters(), lr=config.getfloat('PARAMETERS', 'lr'), weight_decay=config.getfloat('PARAMETERS', 'weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.getint('PARAMETERS', 'step_size'), gamma=config.getfloat('PARAMETERS', 'gamma'))

    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0

    num_epochs = config.getint('PARAMETERS', 'num_epochs')
    patience = config.getint('PARAMETERS', 'patience')
    tic = time.time()
    for e in tqdm(range(1, num_epochs + 1)):
        epoch_tic = time.time()
        report = train(net, loss_fn, optimizer, train_loader, device)
        H['train_loss'].append(report['Loss'].mean())
        H['train_dice'].append(report['Dice'].mean())

        report = validate(net, loss_fn, val_loader, device)
        H['val_loss'].append(report['Loss'].mean())
        H['val_dice'].append(report['Dice'].mean())

        report = test(net, test_loader, device)
        H['test_dice'].append(report['Dice'].mean())

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
        scheduler.step()

        logger.info(AsciiTable([
            ['Split', 'Loss', 'Dice'],
            ['Train', '{:.3f}'.format(H['train_loss'][-1]), '{:.3f}'.format(H['train_dice'][-1])],
            ['Val', '{:.3f}'.format(H['val_loss'][-1]), '{:.3f}'.format(H['val_dice'][-1])],
            ['Test', '-', '{:.3f}'.format(H['test_dice'][-1])],
            ['Epoch', e, epochs_since_last_improvement]
        ]).table)

        if H['val_dice'][-1] > best_dice:
            best_dice = H['val_dice'][-1]
            epochs_since_last_improvement = 0
            torch.save({'epoch': e,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': H['train_loss'][-1],
                        'train_acc': H['train_dice'][-1],
                        'val_loss': H['val_loss'][-1],
                        'val_acc': H['val_dice'][-1],
                        'test_acc': H['test_dice'][-1]
                        }, osp.join(save_dir, 'checkpoint.pth'))
            logger.info(f'Checkpoint updated with dice: {best_dice:,.3f}')
        else:
            epochs_since_last_improvement += 1

        writer.add_scalars('loss', {'train': H['train_loss'][-1], 'val': H['val_loss'][-1]}, e)
        writer.add_scalars('dice', {'train': H['train_dice'][-1], 'val': H['val_dice'][-1], 'test': H['test_dice'][-1]}, e)
        writer.add_scalar('epoch_time', time.time() - epoch_tic, e)

        # early stop
        if epochs_since_last_improvement > patience:
            logger.info(f'Early stop at epoch: {e}')
            break

    logger.info('\nTraining time: {:.3f} hrs.'.format((time.time() - tic) / 3600.0))

    # Plot loss history
    plt.figure()
    plt.plot(H['train_loss'], label='train')
    plt.plot(H['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(osp.join(save_dir, 'loss.png'))

    # Plot dice history
    plt.figure()
    plt.plot(H['train_dice'], label='train')
    plt.plot(H['val_dice'], label='val')
    plt.plot(H['test_dice'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend(loc='lower right')
    plt.savefig(osp.join(save_dir, 'dice.png'))

    writer.close()
