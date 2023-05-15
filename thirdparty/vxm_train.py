import os
import os.path as osp
import sys
import configparser
from terminaltables import AsciiTable
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
from tabulate import tabulate
import pandas as pd
import time
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
import matplotlib.pyplot as plt

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.voxelmorph as vxm  # nopep8

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils as fpu
from utilities import stuff
from datasets.flowunet_dataset import FlowUNetDataset, get_bounds
import segmentation.transforms as T
# from segmentation.utils import create_checkpoint
from utilities.parser_conversions import str_to_tuple
from utilities.fold import create_5fold
from utilities import stuff


def get_fold(data_cfg):
    root_dir = data_cfg['root_dir']
    fold = data_cfg.getint('fold')
    dataset = FlowUNetDataset(root_dir, mode='train')
    return create_5fold(len(dataset))[fold], fold


@torch.no_grad()
def propagate_label(model, img, label, es, ed, fwd, compute_hd, is_test=False):
    *_, mi, mf, indices = get_bounds(es, ed, label, fwd=fwd, test=is_test)
    mi.unsqueeze_(0)
    mf.unsqueeze_(0)
    propagated_mask = mi.clone()

    if is_test:
        est_masks = []

    for i in range(len(indices) - 1):
        tm = indices[i].item()
        tf = indices[i + 1].item()
        moving = img[tm].unsqueeze(0)
        fixed = img[tf].unsqueeze(0)

        # Compute flow between two consecutive frames
        _, flow = model(moving, fixed, registration=True)

        # Propagate masks
        propagated_mask = model.transformer(propagated_mask, flow)

        if is_test:
            est_masks.append(propagated_mask)

    if is_test:
        est_masks = torch.cat(est_masks).round()
        true_masks = label[indices[1:]]
        est_masks = T.one_hot(est_masks, true_masks.shape[1], argmax=True)
        dice = compute_dice(est_masks, true_masks, include_background=False).mean()
        return (dice, compute_hausdorff_distance(est_masks, true_masks, include_background=False).mean()) if compute_hd else dice
    else:
        est_masks = T.one_hot(propagated_mask, mf.shape[1], argmax=True)
        dice = compute_dice(est_masks, mf, include_background=False).mean()
        return (dice, compute_hausdorff_distance(est_masks, mf, include_background=False).mean()) if compute_hd else dice


def train(train_loader, model, optimizer, losses, weights, device, compute_hd, test=False):
    model.train()
    report = pd.DataFrame(columns=['Loss', 'Dice']) if not compute_hd else pd.DataFrame(columns=['Loss', 'Dice', 'HD'])

    for data in train_loader:
        img = data['image'].squeeze(0).to(device)
        label = data['label'].squeeze(0).to(device)

        m_idx = torch.arange(img.shape[0] - 1)
        f_idx = torch.arange(1, img.shape[0])
        moving = img[m_idx]
        fixed = img[f_idx]

        y_pred = model(moving, fixed)

        # calculate total loss
        y_true = [fixed, None]
        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss += curr_loss

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics = propagate_label(model, img, label, data['es'][0], data['ed'][0], fwd=True, compute_hd=compute_hd, is_test=test)
            report.loc[len(report)] = [loss.item(), metrics.item()] if not compute_hd else [loss.item(), metrics[0].item(), metrics[1].item()]
    return report


@torch.no_grad()
def validate(val_loader, model, losses, weights, device):
    model.eval()
    report = pd.DataFrame(columns=['Loss', 'Dice'])

    for data in val_loader:
        img = data['image'].squeeze(0).to(device)
        label = data['label'].squeeze(0).to(device)

        m_idx = torch.arange(img.shape[0] - 1)
        f_idx = torch.arange(1, img.shape[0])
        moving = img[m_idx]
        fixed = img[f_idx]

        y_pred = model(moving, fixed)

        # calculate total loss
        y_true = [fixed, None]
        loss = 0
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss += curr_loss

        dice = propagate_label(model, img, label, data['es'][0], data['ed'][0], fwd=True, compute_hd=False)
        report.loc[len(report)] = [loss.item(), dice.item()]
    return report


@torch.no_grad()
def test(test_loader, model, device, logger=None, verbose=False):
    model.eval()
    report = pd.DataFrame(columns=['Patient', 'Dice', 'HD', 'Dice_Fwd', 'HD_Fwd', 'Dice_Bwd', 'HD_Bwd', 'Time'])

    for data in test_loader:
        img = data['image'].squeeze(0).to(device)
        label = data['label'].squeeze(0).to(device)
        patient = data['patient'][0]
        tic = time.time()
        dice_fwd, hd_fwd = propagate_label(model, img, label, data['es'][0], data['ed'][0], fwd=True, compute_hd=True, is_test=True)
        dice_bwd, hd_bwd = propagate_label(model, img, label, data['es'][0], data['ed'][0], fwd=False, compute_hd=True, is_test=True)
        report.loc[len(report)] = [
            patient,
            (dice_fwd.item() + dice_bwd.item()) / 2,
            (hd_fwd.item() + hd_bwd.item()) / 2,
            dice_fwd.item(),
            hd_fwd.item(),
            dice_bwd.item(),
            hd_bwd.item(),
            time.time() - tic
        ]

    if verbose:
        logger.info(tabulate(report.round(3), headers='keys', tablefmt='psql'))
    return report


def create_checkpoint(model, optimizer, H, e, file):
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': H['train_loss'][-1],
        'train_acc': H['train_dice'][-1],
        'val_loss': H['val_loss'][-1],
        'val_acc': H['val_dice'][-1],
        'test_acc': H['test_dice'][-1]
    }, file)


if __name__ == '__main__':
    # Seeding for reproducible results
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read configuration options
    config = configparser.ConfigParser()
    config.read('parser/vxm_train.ini')
    data_cfg = config['DATA']
    params_cfg = config['PARAMETERS']

    fold, n = get_fold(data_cfg)
    save_dir = fpu.create_save_dir(data_cfg['output_dir'], f'fold_{n}')
    stuff.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    logger.info(f'Save dir: {save_dir}')
    logger.info(f'Device: {device}')
    logger.info(f'Fold: {n}')

    # Create dataloaders
    train_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.OneHotEncoding(data_cfg.getint('num_classes'), keys=['label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    val_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.OneHotEncoding(data_cfg.getint('num_classes'), keys=['label']),
        T.ToTensor(keys=['image', 'label'])
    ])

    train_ds = FlowUNetDataset(data_cfg['root_dir'], 'train', train_transforms, fold_indices=fold['train'])
    val_ds = FlowUNetDataset(data_cfg['root_dir'], 'train', val_transforms, fold_indices=fold['val'])
    test_ds = FlowUNetDataset(data_cfg['root_dir'], 'test', val_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=params_cfg.getint('num_workers'))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=params_cfg.getint('num_workers'))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params_cfg.getint('num_workers'))

    model = vxm.networks.VxmDense(
        inshape=(params_cfg.getint('img_size'),) * 3,
        nb_unet_features=[str_to_tuple(params_cfg['enc_feat'], int), str_to_tuple(params_cfg['dec_feat'], int)],
        bidir=params_cfg.getboolean('bidir'),
        int_steps=params_cfg.getint('int_steps'),
        int_downsize=params_cfg.getint('int_downsize')
    )
    model.to(device)
    stuff.save_model(model, save_dir, 'net.txt')
    optimizer = torch.optim.Adam(model.parameters(), lr=params_cfg.getfloat('lr'))

    if params_cfg['img_loss'] == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif params_cfg['img_loss'] == 'mse':
        image_loss_func = vxm.losses.MSE().loss
        # image_loss_func = nn.MSELoss(reduction='sum')
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % params_cfg['img_loss'])

    losses = [image_loss_func, vxm.losses.Grad('l2', loss_mult=2).loss]
    weights = [1.0, params_cfg.getfloat('lambda')]
    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0
    patience = params_cfg.getint('patience')
    tic = time.time()

    for e in tqdm(range(1, params_cfg.getint('num_epochs') + 1)):
        epoch_tic = time.time()
        report = train(train_loader, model, optimizer, losses, weights, device, False)
        H['train_loss'].append(report['Loss'].mean())
        H['train_dice'].append(report['Dice'].mean())

        report = validate(val_loader, model, losses, weights, device)
        H['val_loss'].append(report['Loss'].mean())
        H['val_dice'].append(report['Dice'].mean())

        report = test(test_loader, model, device, logger, verbose=True)
        H['test_dice'].append(report['Dice'].mean())

        logger.info(AsciiTable([
            ['Split', 'Loss', 'Dice'],
            ['Train', '{:.6f}'.format(H['train_loss'][-1]), '{:.3f}'.format(H['train_dice'][-1])],
            ['Val', '{:.6f}'.format(H['val_loss'][-1]), '{:.3f}'.format(H['val_dice'][-1])],
            ['Test', '-', '{:.3f}'.format(H['test_dice'][-1])],
            ['Epoch', e, epochs_since_last_improvement]
        ]).table)

        if H['val_dice'][-1] > best_dice:
            best_dice = H['val_dice'][-1]
            epochs_since_last_improvement = 0
            create_checkpoint(model, optimizer, H, e, osp.join(save_dir, 'checkpoint_best.pth'))
            model.save(osp.join(save_dir, 'model_best.pt'))
            logger.info(f'Checkpoint updated with dice: {best_dice:,.3f}')

        else:
            epochs_since_last_improvement += 1

        writer.add_scalars('loss', {'train': H['train_loss'][-1], 'val': H['val_loss'][-1]}, e)
        writer.add_scalars('dice', {'train': H['train_dice'][-1], 'val': H['val_dice'][-1], 'test': H['test_dice'][-1]}, e)
        writer.add_scalar('epoch_time', time.time() - epoch_tic, e)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)

        # early stop
        if epochs_since_last_improvement > patience:
            logger.info(f'Early stop at epoch: {e}')
            break

    create_checkpoint(model, optimizer, H, e, osp.join(save_dir, 'checkpoint_final.pth'))
    model.save(osp.join(save_dir, 'model_final.pt'))
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
