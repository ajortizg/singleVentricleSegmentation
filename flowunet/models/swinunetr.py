import os.path as osp
import sys
from configparser import ConfigParser
from monai.networks.nets import SwinUNETR
from torch import nn
import torch

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities.parser_conversions import str_to_tuple


class SwinUNetr(nn.Module):
    def __init__(self, n_classes: int, cfg: ConfigParser):
        super().__init__()
        params = cfg['PARAMETERS']
        n_classes = n_classes + 1 if n_classes > 1 else n_classes
        in_channels = n_classes + 1
        img_size = params.getint('img_size')
        depths = str_to_tuple(params['swin_depths'], int)
        num_heads = str_to_tuple(params['swin_num_heads'], int)
        feature_size = params.getint('swin_feature_size')

        self.net = SwinUNETR(img_size=(img_size, img_size, img_size),
                             in_channels=in_channels,
                             out_channels=n_classes,
                             depths=depths,
                             num_heads=num_heads,
                             feature_size=feature_size,
                             norm_name='instance',
                             spatial_dims=3)

    def forward(self, x):
        identity = x[:, 1:]
        output = self.net(x)
        return output + identity, output
