from abc import ABC
from configparser import ConfigParser
from torch.nn import Module

from .unet import UNet


_supported_models = {
    'UNet': UNet
}


class Factory(ABC):

    @staticmethod
    def create(cfg: ConfigParser) -> Module:
        model = cfg['PARAMETERS']['net']
        if model not in _supported_models:
            raise ValueError(f'Model {model} not found!')
        return _supported_models[model](cfg)
