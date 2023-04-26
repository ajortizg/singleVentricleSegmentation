import os.path as osp
import sys
import torch
from torch import nn
from configparser import ConfigParser

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from thirdparty.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from thirdparty.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class TransUNet(nn.Module):
    def __init__(self, cfg: ConfigParser):
        super().__init__()
        params = cfg['PARAMETERS']

        vit_name = params['vit_name']
        vit_patches_size = params.getint('vit_patches_size')
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = params.getint('num_classes')
        config_vit.n_skip = params.getint('n_skip')
        img_size = params.getint('img_size')

        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        self.net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)

    def forward(self, x):
        return self.net(x)
