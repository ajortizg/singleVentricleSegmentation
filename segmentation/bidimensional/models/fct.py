import os.path as osp
import sys
import torch
from torch import nn
from configparser import ConfigParser

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
from thirdparty.FullyConvolutionalTransformer.PyTorch.fct import FCT as _FCT, init_weights


class FCT(nn.Module):
    def __init__(self, cfg: ConfigParser):
        super().__init__()
        params = cfg['PARAMETERS']
        num_classes = params.getint('num_classes')
        img_size = params.getint('img_size')

        self.net = _FCT(num_classes=num_classes, img_size=img_size)
        init_weights(self.net)

    def forward(self, x):
        return self.net(x)
