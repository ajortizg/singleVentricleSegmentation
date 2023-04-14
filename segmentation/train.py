import os.path as osp
import configparser
import time
import sys


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from monai.metrics.meandice import compute_dice
from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from monai.losses import DiceCELoss
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import pandas as pd
from tqdm import tqdm

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.models.model_factory import create_model, save_model
from segmentation.dataset import SVDataset, create_5fold
import segmentation.transforms as T


def get_fold(config):
    root_dir = config.get('DATA', 'BASE_PATH_3D')
    fold = config.getint('DATA', 'fold')
    dataset = SVDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold


def create_dataloaders(config, fold):
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')
    num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
    data_aug = config['DATA_AUGMENTATION']

    train_transforms = T.Compose([
        T.ToRAS(),
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.RandomRotate(p=data_aug.getfloat('ROT_PROB'),
                       range_z=tuple(map(float, data_aug['ROT_Z_RANGE'].split(','))),
                       range_y=tuple(map(float, data_aug['ROT_Y_RANGE'].split(','))),
                       range_x=tuple(map(float, data_aug['ROT_X_RANGE'].split(','))),
                       boundary=data_aug['ROT_BOUNDARY']),
        T.RandomVerticalFlip(data_aug.getfloat('VERTICAL_FLIP_PROB')),
        T.RandomHorizontalFlip(data_aug.getfloat('HORIZONTAL_FLIP_PROB')),
        T.RandomDepthFlip(data_aug.getfloat('DEPTH_FLIP_PROB')),
        T.ElasticDeformation(p=data_aug.getfloat('ED_PROB'),
                             sigma_range=tuple(map(float, data_aug['ED_SIGMA_RANGE'].split(','))),
                             points=data_aug.getint('ED_GRID'),
                             boundary=data_aug['ED_BOUNDARY'],
                             prefilter=data_aug.getboolean('ED_USE_PREFILTER'),
                             axis=data_aug['ED_AXIS'],
                             order=data_aug.getint('ED_ORDER')),
        # T.GammaScaling(data_aug.getfloat('GAMMA_SCALING_PROB'), tuple(map(float, data_aug['GAMMA_SCALING_RANGE'].split(',')))),
        T.GammaTransform(0.1, (0.7, 1.5), True, False),
        T.GammaTransform(0.3, (0.7, 1.5), False, False),
        T.MutiplicativeScaling(data_aug.getfloat('MULT_SCALING_PROB'),
                               tuple(map(float, data_aug['MULT_SCALING_RANGE'].split(',')))),
        T.AdditiveScaling(data_aug.getfloat('ADD_SCALING_PROB'),
                          data_aug.getfloat('ADD_SCALING_MEAN'),
                          data_aug.getfloat('ADD_SCALING_STD')),
        T.AdditiveGaussianNoise(data_aug.getfloat('NOISE_PROB'),
                                data_aug.getfloat('NOISE_MU'),
                                data_aug.getfloat('NOISE_STD')),
        T.QuadraticNormalization(mean_inside_mask=True),
        T.Discretize(th=0.5),
        T.AddChannelDim(),
        T.OneHotEncoding(num_classes),
        T.ToTensor()
    ])

    val_transforms = T.Compose([
        T.ToRAS(),
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.QuadraticNormalization(mean_inside_mask=True),
        T.Discretize(th=0.5),
        T.AddChannelDim(),
        T.OneHotEncoding(num_classes),
        T.ToTensor()
    ])

    root_dir = config.get('DATA', 'BASE_PATH_3D')
    batch_size = config.getint('PARAMETERS', 'BATCH_SIZE')
    workers = config.getint('PARAMETERS', 'NUM_WORKERS')

    train_ds = SVDataset(root_dir, mode='train', transforms=train_transforms, fold_indices=fold['train'])
    val_ds = SVDataset(root_dir, mode='train', transforms=val_transforms, fold_indices=fold['val'])
    test_ds = SVDataset(root_dir, mode='test', transforms=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=SVDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=SVDataset.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=SVDataset.collate_fn)
    return train_loader, val_loader, test_loader


def train(net, loss_fn, opt, loader, device):
    net.train()
    report = pd.DataFrame(columns=['Loss', 'Dice'])

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            logits, _ = net(img[..., k])
            loss = loss_fn(logits, mask[..., k])
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = torch.softmax(logits, dim=1)
                pred = torch.where(pred > 0.5, 1.0, 0.0)
                dice = compute_dice(pred, mask[..., k], include_background=False).mean()
                report.loc[len(report)] = [loss.item(), dice.item()]
    return report


@torch.no_grad()
def validate(net, loss_fn, loader, device):
    net.eval()
    report = pd.DataFrame(columns=['Loss', 'Dice'])

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            logits, _ = net(img[..., k])
            loss = loss_fn(logits, mask[..., k])

            pred = torch.softmax(logits, dim=1)
            pred = torch.where(pred > 0.5, 1.0, 0.0)
            dice = compute_dice(pred, mask[..., k], include_background=False).mean()
            report.loc[len(report)] = [loss.item(), dice.item()]
    return report


@torch.no_grad()
def test(net, loader, device):
    net.eval()
    report = pd.DataFrame(columns=['Dice'])

    for data in loader:
        img = data['img'].to(device).squeeze(0)
        mask = data['mask'].to(device).squeeze(0)
        img = torch.permute(img, (4, 0, 1, 2, 3))
        mask = torch.permute(mask, (4, 0, 1, 2, 3))

        logits, _ = net(img)
        pred = torch.softmax(logits, dim=1)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        dice = compute_dice(pred, mask, include_background=False).mean()
        report.loc[len(report)] = [dice.item()]
    return report


if __name__ == '__main__':
    plots.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/seg_train.ini')

    fold, n = get_fold(config)
    train_loader, val_loader, test_loader = create_dataloaders(config, fold)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), f'SEG-{n}')
    plots.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = plots.create_logger(save_dir)
    plots.save_json([fold], osp.join(save_dir, f'fold.json'), default=int)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))

    net = create_model(config, logger).to(device)
    save_model(net, save_dir, 'net.txt')
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = DiceCELoss(softmax=True, lambda_dice=0.4, lambda_ce=0.6)
    optimizer = optim.Adam(net.parameters(), lr=config.getfloat('PARAMETERS', 'LR'), weight_decay=config.getfloat('PARAMETERS', 'WEIGHT_DECAY'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.getint('PARAMETERS', 'STEP_SIZE'), gamma=config.getfloat('PARAMETERS', 'GAMMA'))

    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0

    num_epochs = config.getint('PARAMETERS', 'NUM_EPOCHS')
    patience = config.getint('PARAMETERS', 'PATIENCE')
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
