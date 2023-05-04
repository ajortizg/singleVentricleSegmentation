"""Adapted from <https://github.com/annikabrundyn> and <https://github.com/akshaykvnit>"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.unet import UNet
from configparser import ConfigParser
import os
import sys
import os.path as osp


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities.parser_conversions import str_to_tuple


class UNet(nn.Module):
    def __init__(self, n_classes: int, cfg: ConfigParser):
        params = cfg['PARAMETERS']
        n_layers = params.getint('num_layers')
        # Add background class for multiclass approach
        n_classes = n_classes + 1 if n_classes > 1 else n_classes
        in_channels = n_classes + 1
        feats_start = params.getint('features_start')
        trilinear = params.getboolean('trilinear')
        padding = params.getint('padding')
        kernel_size = str_to_tuple(params['kernel_size'], int)
        act = params['activation']
        slope = params.getfloat('activation_slope')

        if n_layers < 1:
            raise ValueError(
                f"Num_layers = {n_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = n_layers
        layers = [DoubleConv3d(in_channels, feats_start, kernel_size, padding, act, slope)]
        feats = feats_start

        for _ in range(n_layers - 1):
            layers.append(Down3d(feats, feats * 2, kernel_size, padding, act, slope))
            feats *= 2

        for _ in range(n_layers - 1):
            layers.append(Up3d(feats, feats // 2, trilinear, kernel_size, padding, act, slope))
            feats //= 2

        out_conv = nn.Conv3d(feats, n_classes, kernel_size=1)
        layers.append(out_conv)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # TODO: it is clone necessary?
        identity = x[:, 1:]

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        output = self.layers[-1](xi[-1])
        return output + identity, output


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, act='relu', slope=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                                 nn.InstanceNorm3d(out_ch),
                                 self.activation(act, slope),
                                 nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                                 nn.InstanceNorm3d(out_ch),
                                 self.activation(act, slope))

    def activation(self, act, slope):
        act_fn = nn.Module
        if act == 'relu':
            act_fn = nn.ReLU()
        elif act == 'leaky_relu':
            act_fn = nn.LeakyReLU(negative_slope=slope)
        elif act == 'prelu':
            act_fn = nn.PReLU(num_parameters=1)
        return act_fn

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, act='relu', slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3d(in_ch, out_ch, kernel_size, padding, act, slope)
        )

    def forward(self, x):
        return self.net(x)


class Up3d(nn.Module):
    """Upsampling (by either trilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv3D."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = False, kernel_size=(3, 3), padding=1, act='relu', slope=0.2):
        super().__init__()
        self.upsample = None
        if trilinear:
            conv = nn.Conv3d(in_ch, in_ch // 2, kernel_size=1)
            ups = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

            self.upsample = nn.Sequential(ups, conv)
        else:
            conv_trans = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.upsample = conv_trans

        self.conv = DoubleConv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, act=act, slope=slope)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_d = x2.shape[2] - x1.shape[2]
        diff_h = x2.shape[3] - x1.shape[3]
        diff_w = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
