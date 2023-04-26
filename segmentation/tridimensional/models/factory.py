from enum import Enum
from abc import ABC
from configparser import ConfigParser

from .unet import UNet
from .swin_unetr import SwinUNetr
from .unetr import UNetr
from .resunet import ResUNet
from torch.nn import Module


_supported_models = {
    'UNet': UNet,
    'ResUNet': ResUNet,
    'UNetr': UNetr,
    'SwinUNetr': SwinUNetr
}


class Factory(ABC):

    @staticmethod
    def create(cfg: ConfigParser) -> Module:
        model = cfg['PARAMETERS']['net']
        if model not in _supported_models:
            raise ValueError(f'Model {model} not found!')
        return _supported_models[model](cfg)
