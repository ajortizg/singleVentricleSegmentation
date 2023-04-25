import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from monai.metrics.meandice import compute_dice
from monai.losses.dice import DiceCELoss, DiceLoss
from terminaltables import AsciiTable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from FullyConvolutionalTransformer.PyTorch.fct import FCT, init_weights
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from fct_trainer import FCTTrainer
from transunet_trainer import TransUNetTrainer
from MERIT.lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.transforms as T
from utilities.fold import create_5fold
from utilities import stuff
import utilities.file_paths_utils as fpu
import segmentation.utils as utils
from cnn.models.model_factory import save_model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = 'data/Dataset_SVD_crop_2d'
    num_classes = 2
    img_size = 96
    fold_n = 0
    lr = 1e-3
    epochs = 500
    patience = 50
    batch_size = 16
    num_workers = 16

    dataset = SegmentationDataset(root_dir, mode='train')
    fold = create_5fold(len(dataset))[fold_n]
    save_dir = fpu.create_save_dir('results/FCT_results/Dataset_SVD_crop_2d', f'fold_{fold_n}')

    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    fpu.save_json([fold], osp.join(save_dir, 'fold.json'), default=int)
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))
    logger.info('Fold: {}'.format(fold_n))

    train_transforms = T.Compose([
        T.AddNLeadingDims(n=3, keys=['image', 'label']),
        T.RandomFlip(0.5, axis=3, keys=['image', 'label']),
        T.RandomFlip(0.5, axis=4, keys=['image', 'label']),
        T.RandomRotate2D(0.3, (0, 360)),
        T.ElasticDeformation(0.15, (0.5, 2.0), 8, 'constant', 'yx'),
        T.SimulateLowResolution(0.25, (0.8, 1.0)),
        T.GammaCorrection(0.5, (0.7, 1.5), retain_stats=True, invert_image=False),
        T.GammaCorrection(0.1, (0.7, 1.5), retain_stats=True, invert_image=True),
        T.MultiplicativeScaling(0.15, (0.75, 1.25)),
        T.AdditiveGaussianNoise(0.1, (0.0, 0.1)),
        T.ContrastAugmentation(0.15, (0.75, 1.25)),
        T.GaussialBlur(0.2, (0.5, 1.0)),
        T.RemoveNLeadingDims(n=2, keys=['image', 'label']),
        T.ToTensor(keys=['image', 'label'])
    ])
    val_transforms = T.Compose([
        T.AddNLeadingDims(n=1, keys=['image', 'label']),
        T.ToTensor(keys=['image', 'label'])
    ])
    fpu.save_transforms_to_json(train_transforms, osp.join(save_dir, 'transforms.json'))

    train_ds = SegmentationDataset(root_dir, mode='train', transforms=train_transforms, fold_idxs=fold['train'])
    val_ds = SegmentationDataset(root_dir, mode='train', transforms=val_transforms, fold_idxs=fold['val'])
    test_ds = SegmentationDataset(root_dir, mode='test', transforms=val_transforms)
    do_test = len(test_ds) > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if do_test:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vit_name = 'R50-ViT-B_16'
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).to(device)

    # model = FCT(num_classes=num_classes, img_size=img_size).to(device)
    # model.apply(init_weights)
    save_model(model, save_dir, 'net.txt')

    loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.3, lambda_dice=0.7)
    with open(osp.join(save_dir, 'loss_fn.txt'), 'w') as f:
        f.write(str(loss_fn.__class__))
        f.write(str(loss_fn.__dict__))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.998)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

    H = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'test_dice': []}
    epochs_since_last_improvement = 0
    best_dice = 0.0

    # trainer = FCTTrainer(img_size, num_classes, model, loss_fn, optimizer, device)
    trainer = TransUNetTrainer(num_classes, model, loss_fn, optimizer, device)

    tic = time.time()
    for e in tqdm(range(1, epochs + 1)):
        epoch_tic = time.time()
        report = trainer.train(train_loader)
        H['train_loss'].append(report['Loss'].mean())
        H['train_dice'].append(report['Dice'].mean())

        report = trainer.validate(val_loader)
        H['val_loss'].append(report['Loss'].mean())
        H['val_dice'].append(report['Dice'].mean())

        if do_test:
            report = trainer.test(test_loader)
            H['test_dice'].append(report['Dice'].mean())
        else:
            H['test_dice'].append(0)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
        scheduler.step(H['val_loss'][-1])

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
            utils.create_checkpoint(model, e, optimizer, H, osp.join(save_dir, 'checkpoint_best.pth'))
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

    utils.create_checkpoint(model, e, optimizer, H, osp.join(save_dir, 'checkpoint_final.pth'))
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
