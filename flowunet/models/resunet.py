from configparser import ConfigParser

from monai.networks import nets
from torch import nn


class ResUNet(nn.Module):
    def __init__(self, n_classes: int, cfg: ConfigParser):
        super().__init__()
        params = cfg['PARAMETERS']
        num_layers = params.getint('num_layers')
        n_classes = n_classes + 1 if n_classes > 1 else n_classes
        in_channels = n_classes + 1
        features_start = params.getint('features_start')
        num_res_units = params.getint('num_res_units')

        channels = [features_start]
        strides = []
        for _ in range(1, num_layers):
            channels.append(channels[-1] * 2)
            strides.append(2)

        self.net = nets.UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=n_classes,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

    def forward(self, x):
        identity = x[:, 1:]
        output = self.net(x)
        return output + identity, output
