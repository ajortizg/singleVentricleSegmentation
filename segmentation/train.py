from dataset import SVDSegmentation
import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch
import os.path as osp
import configparser
import time
import sys
from monai.metrics.meandice import compute_dice
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import numpy as np
from torch.utils.tensorboard import SummaryWriter


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots
from cnn.models.model_factory import create_model, save_model


def create_dataloaders(config):
    img_sz = config.getint('PARAMETERS', 'IMG_SIZE')

    train_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.RandomRotate(p=config.getfloat('DATA_AUGMENTATION', 'ROT_PROB'),
                       range_z=tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Z_RANGE').split(','))),
                       range_y=tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_Y_RANGE').split(','))),
                       range_x=tuple(map(float, config.get('DATA_AUGMENTATION', 'ROT_X_RANGE').split(','))),
                       boundary=config.get('DATA_AUGMENTATION', 'ROT_BOUNDARY')),
        T.RandomVerticalFlip(config.getfloat('DATA_AUGMENTATION', 'VERTICAL_FLIP_PROB')),
        T.RandomHorizontalFlip(config.getfloat('DATA_AUGMENTATION', 'HORIZONTAL_FLIP_PROB')),
        T.RandomDepthFlip(config.getfloat('DATA_AUGMENTATION', 'DEPTH_FLIP_PROB')),
        T.ElasticDeformation(p=config.getfloat('DATA_AUGMENTATION', 'ED_PROB'),
                             sigma_range=tuple(map(float, config.get('DATA_AUGMENTATION', 'ED_SIGMA_RANGE').split(','))),
                             points=config.getint('DATA_AUGMENTATION', 'ED_GRID'),
                             boundary=config.get('DATA_AUGMENTATION', 'ED_BOUNDARY'),
                             prefilter=config.getboolean('DATA_AUGMENTATION', 'ED_USE_PREFILTER'),
                             axis=config.get('DATA_AUGMENTATION', 'ED_AXIS'),
                             order=config.getint('DATA_AUGMENTATION', 'ED_ORDER')),
        T.GammaScaling(config.getfloat('DATA_AUGMENTATION', 'GAMMA_SCALING_PROB'),
                       tuple(map(float, config.get('DATA_AUGMENTATION', 'GAMMA_SCALING_RANGE').split(',')))),
        T.MutiplicativeScaling(config.getfloat('DATA_AUGMENTATION', 'MULT_SCALING_PROB'),
                               tuple(map(float, config.get('DATA_AUGMENTATION', 'MULT_SCALING_RANGE').split(',')))),
        T.AdditiveScaling(config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_PROB'),
                          config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_MEAN'),
                          config.getfloat('DATA_AUGMENTATION', 'ADD_SCALING_STD')),
        T.AdditiveGaussianNoise(config.getfloat('DATA_AUGMENTATION', 'NOISE_PROB'),
                                config.getfloat('DATA_AUGMENTATION', 'NOISE_MU'),
                                config.getfloat('DATA_AUGMENTATION', 'NOISE_STD')),
        T.QuadraticNormalization(p=1.0),
        T.BinarizeMasks(th=0.5),
        T.ToTensor(add_ch_dim=True)
    ])

    val_transforms = T.Compose([
        T.CropForeground(p=1.0, tol=10),
        T.Resize(p=1.0, size=(img_sz, img_sz, img_sz)),
        T.QuadraticNormalization(p=1.0),
        T.BinarizeMasks(th=0.5),
        T.ToTensor(add_ch_dim=True)
    ])

    root_dir = config.get('DATA', 'BASE_PATH_3D')
    batch_size = config.getint('PARAMETERS', 'BATCH_SIZE')
    workers = config.getint('PARAMETERS', 'NUM_WORKERS')

    train_ds = SVDSegmentation(root_dir, mode='train', transforms=train_transforms)
    val_ds = SVDSegmentation(root_dir, mode='val', transforms=val_transforms)
    test_ds = SVDSegmentation(root_dir, mode='test', transforms=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=SVDSegmentation.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=SVDSegmentation.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=SVDSegmentation.collate_fn)
    return train_loader, val_loader, test_loader


def train(net, loss_fn, opt, loader, device):
    net.train()
    loss_list = []
    dice_list = []

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            logits, _ = net(img[..., k])
            loss = loss_fn(logits, mask[..., k])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_list.append(loss.item())

            with torch.no_grad():
                pred = torch.where(torch.sigmoid(logits) > 0.5, 1.0, 0.0)
                dice = compute_dice(pred, mask[..., k]).mean()
                dice_list.append(dice.item())

    return np.array(loss_list).mean(), np.array(dice_list).mean()


@torch.no_grad()
def validate(net, loss_fn, loader, device):
    net.eval()
    loss_list = []
    dice_list = []

    for data in loader:
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        for k in range(2):
            logits, _ = net(img[..., k])
            loss = loss_fn(logits, mask[..., k])
            loss_list.append(loss.item())

            pred = torch.where(torch.sigmoid(logits) > 0.5, 1.0, 0.0)
            dice = compute_dice(pred, mask[..., k]).mean()
            dice_list.append(dice.item())

    return np.array(loss_list).mean(), np.array(dice_list).mean()


@torch.no_grad()
def test(net, loader, device):
    net.eval()
    dice_list = []

    for data in loader:
        img = data['img'].to(device).squeeze(0)
        mask = data['mask'].to(device).squeeze(0)
        img = torch.permute(img, (4, 0, 1, 2, 3))
        mask = torch.permute(mask, (4, 0, 1, 2, 3))

        logits, _ = net(img)
        pred = torch.where(F.sigmoid(logits) > 0.5, 1.0, 0.0)
        dice = compute_dice(pred, mask).mean()
        dice_list.append(dice.item())

    return np.array(dice_list).mean()


if __name__ == '__main__':
    plots.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/train_segmentation.ini')

    train_loader, val_loader, test_loader = create_dataloaders(config)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'SEG')
    plots.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = plots.create_logger(save_dir)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))

    net = create_model(config, logger).to(device)
    save_model(net, save_dir, 'net.txt')
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.getfloat('PARAMETERS', 'LR'), weight_decay=config.getfloat('PARAMETERS', 'WEIGHT_DECAY'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.getint('PARAMETERS', 'STEP_SIZE'), gamma=config.getfloat('PARAMETERS', 'GAMMA'))

    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0

    num_epochs = config.getint('PARAMETERS', 'NUM_EPOCHS')
    patience = config.getint('PARAMETERS', 'PATIENCE')
    tic = time.time()
    for e in tqdm(range(1, num_epochs + 1)):
        loss, dice = train(net, loss_fn, optimizer, train_loader, device)
        H['train_loss'].append(loss)
        H['train_dice'].append(dice)

        loss, dice = validate(net, loss_fn, val_loader, device)
        H['val_loss'].append(loss)
        H['val_dice'].append(dice)

        dice = test(net, test_loader, device)
        H['test_dice'].append(dice)

        scheduler.step()

        logger.info(AsciiTable([
            ['Split', 'Loss', 'Dice'],
            ['Train', '{:.3f}'.format(H['train_loss'][-1]), '{:.3f}'.format(H['train_dice'][-1])],
            ['Val', '{:.3f}'.format(H['val_loss'][-1]), '{:.3f}'.format(H['val_dice'][-1])],
            ['Test', '-', '{:.3f}'.format(H['test_dice'][-1])],
            ['RPD', '{:.3f}'.format(abs(H['train_loss'][-1] - H['val_loss'][-1]) / ((H['train_loss'][-1] + H['val_loss'][-1]) / 2)),
             '{:.3f}'.format(abs(H['train_dice'][-1] - H['val_dice'][-1]) / ((H['train_dice'][-1] + H['val_dice'][-1]) / 2))],
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
