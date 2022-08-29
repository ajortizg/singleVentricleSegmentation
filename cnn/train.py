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
from trainer import Trainer


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import param_reader
import utils.transforms.senary_transforms as T6
from cnn.dataset import *
from utils.collate import *
from utils import plots


if __name__ == "__main__":
    # plots.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = configparser.ConfigParser()
    config.read('parser/configCNNTrain.ini')

    P = param_reader.train_params(config)

    # Create train and validation datasets
    train_transforms = T6.Compose([
        T6.RandomRotate(P['rot_prob'], P['rot_range_x'], P['rot_range_y'], P['rot_range_z'], boundary=P['rot_boundary'], clip_interval=P['clip_interval']),
        T6.ElasticDeformation(P['ed_prob'], P['ed_sigma_range'], P['ed_grid'], P['ed_boundary'], P['ed_prefilter'], P['ed_axis'], P['clip_interval']),
        T6.RandomVerticalFlip(P['vflip_prob']),
        T6.RandomHorizontalFlip(P['hflip_prob']),
        T6.RandomDepthFlip(P['dflip_prob']),
        T6.GammaScaling(P['gamma_scaling_prob'], P['gamma_scaling_range']),
        T6.MutiplicativeScaling(P['mult_scaling_prob'], P['gamma_scaling_range'], P['clip_interval']),
        T6.OneOf([
            T6.AdditiveScaling(P['add_scaling_prob'], P['add_scaling_mean'], P['add_scaling_std'], P['clip_interval']),
            T6.AdditiveGaussianNoise(P['noise_prob'], P['noise_mu'], P['noise_std'], P['clip_interval'])
        ]),
        T6.BinarizeMasks(th=0.5),
        T6.ToTensor()
    ])

    val_transforms = T6.Compose([T6.ToTensor()])

    train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.ED_ES, full_transforms=train_transforms)
    val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.ED_ES, full_transforms=val_transforms)
    test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.WHOLE_CYCLE, full_transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=P['batch_size'], shuffle=True, num_workers=P['workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=P['batch_size'], shuffle=False, num_workers=P['workers'], collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=collate_fn)

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'CNN')
    logger = plots.create_logger(save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger.info(f'Using device {device}')

    # Create model
    net = create_model(config, logger).to(device)
    summary(net, input_size=(2, 80, 80, 80), batch_size=P['batch_size'])
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
    trainer = Trainer(net, opt, pbar, config, device, writer, logger)
    tic = time.time()

    for e in range(P['epochs']):
        trainer.train_epoch(train_loader)
        trainer.val_epoch(val_loader)
        trainer.test_epoch(test_loader, test_ds)

        trainer.log(e)
        trainer.create_checkpoint(e, save_dir, when_better=True, verbose=True)

        scheduler.step()
        pbar.update(1)

        # early stop
        if trainer.epochs_since_last_improvement > P['patience']:
            logger.info(f'Early stop at epoch: {e}')
            break

    toc = time.time()
    logger.info('\nTotal time taken to train the model: {:.3f}s'.format(toc - tic))

    trainer.create_checkpoint(e, save_dir, when_better=False)
    trainer.save_stats(save_dir)
    trainer.plot_stats(save_dir, log_scale=True)

    pbar.close()
    writer.close()
