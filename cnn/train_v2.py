import sys
import os.path as osp
import os
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils as fpu
from utilities import stuff
from datasets.flow_unet_dataset import FlowUNetDataset
import segmentation.transforms as T
from cnn.trainer_onehot_v2 import Trainer
from ofunet.models.factory import Factory
from cnn.loss import CustomLoss

if __name__ == '__main__':
    stuff.seeding(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = ConfigParser()
    config.read('parser/ofunet_train.ini')
    data = config['DATA']
    params = config['PARAMETERS']

    # Create dataset
    transforms = T.Compose([
        T.XYZT_To_TZYX(keys=['image', 'label']),
        T.AddDimAt(axis=1, keys=['image', 'label']),
        T.Flow_T3XYZ_To_T3ZYX(keys=['forward_flow', 'backward_flow']),
        # T.RandomFlip(p=1.0, axis=2, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # T.RandomFlip(p=1.0, axis=3, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # T.RandomFlip(p=1.0, axis=4, keys=['image', 'label', 'forward_flow', 'backward_flow']),
        # # T.Resize(1.0, (128, 64, 108), keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.RandomRotate(1.0, (0, 360), (0, 360), (0, 360), 'zeros', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.RandomScale(1.0, (0.5, 1.5), 'zeros', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.ElasticDeformation(1.0, (0.5, 1.5), 8, 'constant', 'yx', keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        # T.SimulateLowResolution(1.0, (0.5, 1.0), keys=['image', 'label', 'forward_flow', 'backward_flow'], label_key='label'),
        T.FlowChannelToLastDim(keys=['forward_flow', 'backward_flow']),
        T.ExtremaPoints(['label']),
        T.ToTensor(keys=['image', 'label', 'forward_flow', 'backward_flow', 'mi', 'mf'])
    ])

    full_ds = FlowUNetDataset(data['root_dir'], 'full', transforms, load_flow=True)
    full_loader = DataLoader(full_ds, batch_size=4, shuffle=True, num_workers=8, collate_fn=FlowUNetDataset.collate)
    n_classes = full_ds.num_classes()

    net = Factory.create(config).to(device)
    gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    net = torch.nn.DataParallel(net, device_ids=np.arange(gpus).tolist())

    loss_fn = CustomLoss(params.getfloat('loss_lambda'), params.getfloat('loss_penalization_gamma'))
    opt = optim.Adam(net.parameters(), lr=params.getfloat('lr'), weight_decay=params.getfloat('weight_decay'))
    trainer = Trainer(config, 2, net, None, None, device, None)

    trainer.train(full_loader)
