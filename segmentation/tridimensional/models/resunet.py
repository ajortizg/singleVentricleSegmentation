from configparser import ConfigParser

from monai.networks import nets
from torch import nn


class ResUNet(nn.Module):
    def __init__(self, cfg: ConfigParser):
        super().__init__()
        num_layers = cfg.getint('PARAMETERS', 'num_layers')
        num_classes = cfg.getint('PARAMETERS', 'num_classes')
        input_channels = cfg.getint('PARAMETERS', 'input_channels')
        features_start = cfg.getint('PARAMETERS', 'features_start')
        num_res_units = cfg.getint('PARAMETERS', 'num_res_units')

        channels = [features_start]
        strides = []
        for _ in range(1, num_layers):
            channels.append(channels[-1] * 2)
            strides.append(2)

        self.unet = nets.UNet(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=num_classes, channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

    def forward(self, x):
        return self.unet(x)
