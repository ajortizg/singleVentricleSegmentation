from enum import Enum
from abc import ABC
from configparser import ConfigParser
import os.path as osp
import sys
from .fct import FCT
from .transunet import TransUNet
from torch import nn
# from MERIT.lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded

_supported_models = {
    'FCT': FCT,
    'TransUNet': TransUNet
}


class Factory(ABC):

    @staticmethod
    def create(cfg: ConfigParser) -> nn.Module:
        model = cfg['PARAMETERS']['net']
        if model not in _supported_models:
            raise ValueError(f'Model {model} not found!')
        return _supported_models[model](cfg)
