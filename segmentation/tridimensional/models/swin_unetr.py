from configparser import ConfigParser
import os.path as osp
import sys

from monai.networks import nets
from torch import nn

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from utilities.parser_conversions import str_to_tuple


class SwinUNetr(nn.Module):
    def __init__(self, cfg: ConfigParser):
        super().__init__()
        param_cfg = cfg['PARAMETERS']
        num_classes = param_cfg.getint('num_classes')
        input_channels = param_cfg.getint('input_channels')
        img_size = param_cfg.getint('img_size')
        depths = str_to_tuple(param_cfg['swin_depths'], int)
        num_heads = str_to_tuple(param_cfg['swin_num_heads'], int)
        feature_size = param_cfg.getint('swin_feature_size')

        self.net = nets.SwinUNETR(img_size=(img_size, img_size, img_size),
                                  in_channels=input_channels,
                                  out_channels=num_classes,
                                  depths=depths,
                                  num_heads=num_heads,
                                  feature_size=feature_size,
                                  norm_name='instance',
                                  spatial_dims=3)

    def forward(self, x):
        return self.net(x)
