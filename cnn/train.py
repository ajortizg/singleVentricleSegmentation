import torch
import configparser
import sys
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os.path as osp
import json
import os
from torchsummary import summary
from models.model_factory import create_model, save_model
# from trainer_single_batch import Trainer
from trainer_multi_batch import Trainer
from trainer_onehot import TrainerOneHot


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import param_reader
import utilities.transforms.senary_transforms as T6
from cnn.dataset import *
from utilities.collate import *
import utilities.path_utils as path_utils
from utilities import stuff
# from utilities.fold import create_5fold


# def get_fold(config):
#     fold = config.getint('DATA', 'fold')
#     dataset = SingleVentricleDataset(config, 'train', LoadFlowMode.NO_LOAD)
#     return create_5fold(len(dataset))[fold], fold


if __name__ == "__main__":
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    P = param_reader.train_params(config)

    # Create train and validation datasets
    train_transforms = T6.Compose([
        T6.RandomRotate(P['rot_prob'], P['rot_range_x'], P['rot_range_y'], P['rot_range_z'], P['rot_boundary']),
        T6.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['ed_order']),
        T6.RandomVerticalFlip(P['vflip_prob']),
        T6.RandomHorizontalFlip(P['hflip_prob']),
        T6.RandomDepthFlip(P['dflip_prob']),
        # T6.Resize(1.0, (64, 64, 64)),  # for swin unet
        # T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
        T6.GammaScaling_V2(P['gamma_scaling_prob'], P['gamma_scaling_range'], P['gamma_invert_image'], P['gamma_retain_stats']),
        T6.ContrastAugmentation(P['contrast_prob'], P['contrast_range'], P['contrast_preserve_range']),
        T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range']),
        T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std']),
        T6.GaussianBlur(P['blur_prob'], P['blur_sigma_range']),
        T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std_range']),
        T6.BinarizeMasks(th=0.5),  # TODO! this only works for svd
        # T6.RoundMasks(),
        T6.ToTensor()
    ])

    val_transforms = T6.Compose([
        # T6.Resize(1.0, (64, 64, 64)),  # for swin unet
        # T6.RoundMasks(), #  # for swin unet
        T6.ToTensor()
    ])
    # fold, n = get_fold(config)

    # train_ds = SingleVentricleDataset(config, 'train', LoadFlowMode.ED_ES, full_transforms=train_transforms, fold_indices=fold['train'])
    # val_ds = SingleVentricleDataset(config, 'train', LoadFlowMode.ED_ES, full_transforms=val_transforms, fold_indices=fold['val'])
    # test_ds = SingleVentricleDataset(config, 'test', LoadFlowMode.WHOLE_CYCLE, full_transforms=val_transforms)
    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=train_transforms)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=val_transforms)
    test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.WHOLE_CYCLE, full_transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=True, num_workers=P['workers'], collate_fn=collate_fn_batch)
    val_loader = DataLoader(val_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['workers'], collate_fn=collate_fn_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=collate_fn_batch)

    save_dir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), f'CNN')
    logger = stuff.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger.info(f'Using device {device}')

    # Create model
    net = create_model(config, logger).to(device)
    # summary(net.net, input_size=(2, 80, 80, 80), batch_size=P['batch_size'])
    net = torch.nn.DataParallel(net, device_ids=np.arange(P['gpus']).tolist())
    if P['pretrained']:
        checkpoint_file = P['checkpoint_file']
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f'Use pretrained weights: {checkpoint_file}')
        opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        opt = optim.Adam(net.parameters(), lr=P['lr'], weight_decay=P['weight_decay'], betas=(P['beta1'], P['beta2']))

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=P['step_size'], gamma=P['gamma'])

    param_reader.save_config(config, save_dir, filename='config.ini')
    save_model(net, save_dir, 'net.txt')

    logger.info('Save directory: ' + save_dir)
    logger.info('Trainig CNN')

    pbar = tqdm(total=P['epochs'])
    # trainer = TrainerOneHot(net, opt, pbar, config, device, writer, logger)
    trainer = Trainer(net, opt, pbar, config, device, writer, logger)
    patience = P['patience']
    tic = time.time()

    for e in range(P['epochs']):
        trainer.train_epoch(train_loader)
        trainer.val_epoch(val_loader)
        trainer.test_epoch(test_loader, test_ds)

        trainer.log(e)
        trainer.create_checkpoint(e, save_dir, when_better=True, which=P['which'], verbose=True)

        scheduler.step()
        pbar.update(1)

        # early stop
        if trainer.epochs_since_last_improvement > patience:
            logger.info(f'Early stop at epoch: {e}')
            break

    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.3f}s'.format(toc - tic))

    trainer.create_checkpoint(e, save_dir, when_better=False)
    trainer.save_stats(save_dir)
    trainer.plot_stats(save_dir, log_scale=True)

    pbar.close()
    writer.close()
