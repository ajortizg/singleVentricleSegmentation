import os.path as osp
import configparser
import sys
import os
from typing import Any

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
import numpy as np


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
import utilities.path_utils as path_utils
from utilities import stuff
from datasets.segmentation_dataset import SegmentationDataset
import segmentation.utils as utils
import segmentation.transforms as T
from segmentation.tridimensional.models.factory import Factory
from segmentation.tridimensional.trainer_3d import FineTuner3d


if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read configuration
    config = configparser.ConfigParser()
    config.read('parser/seg3d_finetuning.ini')
    data = config['DATA']
    params = config['PARAMETERS']

    # Fetch data
    img_sz = params.getint('img_size')
    norm = utils.norm_fn(params['norm'])
    img_key, label_key = 'image', 'label'
    keys = [img_key, label_key]

    test_transforms = T.Compose([
        T.XYZT_To_TZYX(keys=keys),
        T.AddDimAt(axis=1, keys=keys),
        T.ToRAS(keys=keys),
        T.CropForeground(keys=keys, label_key=label_key),
        norm,
        T.Resize(1.0, (img_sz, img_sz, img_sz), keys=keys),
        T.ToTensor(keys=keys)
    ])

    train_ds = SegmentationDataset(data['root_dir'], 'test', test_transforms)
    train_ds.filter_patient(data['patient'])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=params.getint('num_workers'))
    test_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=params.getint('num_workers'))

    # Debug dirs
    save_dir = path_utils.create_save_dir(data['output_dir'], data['patient'])
    stuff.save_config(config, save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    logger = stuff.create_logger(save_dir)
    stuff.save_transforms_to_json(train_ds.transforms, osp.join(save_dir, 'train_transforms.json'))
    logger.info('Save dir: {}'.format(save_dir))
    logger.info('Device: {}'.format(device))

    # Create model
    net = Factory.create(config).to(device)
    # gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # net = torch.nn.DataParallel(net, device_ids=np.arange(gpus).tolist())
    checkpoint = torch.load(data['pretrained_weights'])
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    stuff.save_model(net, save_dir, 'net.txt')

    # Optimization
    loss_fn = loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.getint('step_size'), gamma=params.getfloat('gamma'))

    # Training loop
    trainer = FineTuner3d(params.getint('num_classes'), net, loss_fn, optimizer, device, logger)
    trainer.training_loop(params.getint('num_epochs'),
                          scheduler,
                          train_loader,
                          None,
                          test_loader,
                          writer,
                          save_dir,
                          params.getint('patience'),
                          metric_early_stopping='train_dice',
                          initial_best_metric_value=0,
                          comparisson_fn=lambda metric, best_metric: metric > best_metric)

    trainer.plot_loss_history(osp.join(save_dir, 'loss.png'))
    trainer.plot_accuracy_history(osp.join(save_dir, 'dice.png'))
    writer.close()
