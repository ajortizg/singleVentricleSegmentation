from abc import ABC
from configparser import ConfigParser
from torch.nn import Module

from .unet import UNet
from .swinunetr import SwinUNetr
from .unetr import UNetr
from .resunet import ResUNet


_supported_models = {
    'UNet': UNet,
    'SwinUNetr': SwinUNetr,
    'UNetr': UNetr,
    'ResUNet': ResUNet
}


class Factory(ABC):

    @staticmethod
    def create(n_classes: int, cfg: ConfigParser) -> Module:
        model = cfg['PARAMETERS']['net']
        if model not in _supported_models:
            raise ValueError(f'Model {model} not found!')
        return _supported_models[model](n_classes, cfg)
