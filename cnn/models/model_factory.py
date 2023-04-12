from .basic_unet import *
from .unet_3d import *
from .unet import *
from .unetr import *
from .swin_unetr import *
from torch import nn
import sys
import os.path as osp

__all__ = ['create_model', 'save_model']


def create_model(config, logger=None):
    net_type = config.get('PARAMETERS', 'NET')
    net = nn.Module

    if net_type == 'unet3d':
        net = UNet3d(config, logger)
    elif net_type == 'basic_unet3d':
        net = BasicUNet3d(config, logger)
    elif net_type == 'res_unet3d':
        net = ResUNet3d(config, logger)
    elif net_type == 'unet':
        net = Unet(config, logger)
    elif net_type == 'unetr':
        net = TransformerUNet(config, logger)
    elif net_type == 'swin_unetr':
        net = SwinUNetr(config, logger)
    else:
        print('Unknown network: ' + net_type)
        sys.exit()

    return net


def save_model(net, save_dir, filename):
    filepath = osp.join(save_dir, filename)
    total_params = 0

    modules = [module for module in net.modules()]
    params = [param for param in net.parameters()]
    with open(filepath, 'w') as mfile:
        for idx, m in enumerate(modules):
            mfile.write(f'{idx} -> {m}\n')
        for p in params:
            total_params += p.numel()
        mfile.write(f'\nTotal parameters: {total_params}')
    mfile.close()
