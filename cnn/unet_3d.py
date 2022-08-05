"""Adapted from <https://github.com/annikabrundyn> and <https://github.com/akshaykvnit>"""
import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import lipschitz as L
from monai.networks.nets.basic_unet import BasicUNet
from monai.networks.nets.unet import UNet


class UNet3D(nn.Module):
    def __init__(self, config, logger):
        num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        trilinear = config.getboolean('PARAMETERS', 'TRILINEAR')
        padding = config.getint('PARAMETERS', 'PADDING')
        kstr = config.get('PARAMETERS', 'KERNEL_SIZE')
        kernel_size = tuple(map(int, kstr.split(',')))
        act = config.get('PARAMETERS', 'ACTIVATION')
        slope = config.getfloat('PARAMETERS', 'ACTIVATION_SLOPE')
        lipschitz_reg = config.getboolean('PARAMETERS', 'LIPSCHITZ_REGULARIZATION')
        max_lc = config.getfloat('PARAMETERS', 'MAX_LIPSCHITZ_CONSTANT')
        power_its = config.getint('PARAMETERS', 'POWER_ITS')
        power_eps = config.getfloat('PARAMETERS', 'POWER_EPS')
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')

        if num_layers < 1:
            raise ValueError(
                f"Num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers
        in_size = 80
        layers = [DoubleConv3d(input_channels, features_start, kernel_size, padding, act,
                               slope, lipschitz_reg, max_lc, power_its, power_eps, in_size)]
        feats = features_start

        for _ in range(num_layers - 1):
            layers.append(Down3d(feats, feats * 2, kernel_size, padding, act, slope, lipschitz_reg, max_lc, power_its, power_eps, in_size))
            feats *= 2
            in_size //= 2

        for _ in range(num_layers - 1):
            layers.append(Up3d(feats, feats // 2, trilinear, kernel_size, padding, act, slope, lipschitz_reg, max_lc, power_its, power_eps, in_size))
            feats //= 2
            in_size *= 2

        if lipschitz_reg:
            layers.append(P.register_parametrization(nn.Conv3d(feats, num_classes, kernel_size=1), 'weight',
                                                     L.L2LipschitzConv3d(in_size, ks=1, eps=power_eps, iterations=power_its, max_lc=max_lc)))
        else:
            layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.residual:
            identity = x[:, 1:2, :, :, :].clone()

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        if self.residual:
            return self.layers[-1](xi[-1]) + identity, self.layers[-1](xi[-1])
        else:
            return self.layers[-1](xi[-1]), self.layers[-1](xi[-1])


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, act='relu', slope=0.2,
                 lipschitz=False, max_lc=1.0, power_its=1, power_eps=1e-8, in_size=8):
        super().__init__()
        self.net = nn.Sequential(
            P.register_parametrization(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                'weight', L.L2LipschitzConv3d(
                    in_size, ks=kernel_size, padding=padding, eps=power_eps, iterations=power_its, max_lc=max_lc))
            if lipschitz else nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch),
            self.activation_fn(act, slope),
            P.register_parametrization(
                nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
                'weight', L.L2LipschitzConv3d(
                    in_size, ks=kernel_size, padding=padding, eps=power_eps, iterations=power_its, max_lc=max_lc))
            if lipschitz else nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch),
            self.activation_fn(act, slope))

    def activation_fn(self, act, slope):
        act_fn = nn.Module
        if act == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            act_fn = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif act == 'prelu':
            act_fn = nn.PReLU(num_parameters=1)
        return act_fn

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, act='relu', slope=0.2,
                 lipschitz=False, max_lc=1.0, power_its=1, power_eps=1e-8, in_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3d(in_ch, out_ch, kernel_size, padding, act, slope, lipschitz, max_lc, power_its, power_eps, in_size)
        )

    def forward(self, x):
        return self.net(x)


class Up3d(nn.Module):
    """Upsampling (by either trilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv3D."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = False, kernel_size=(3, 3), padding=1, act='relu', slope=0.2,
                 lipschitz=False, max_lc=1.0, power_its=1, power_eps=1e-8, in_size=8):
        super().__init__()
        self.upsample = None
        if trilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                P.register_parametrization(
                    nn.Conv3d(in_ch, in_ch // 2, kernel_size=1), 'weight',
                    L.L2LipschitzConv3d(in_size, ks=kernel_size, eps=power_eps, iterations=power_its, max_lc=max_lc))
                if lipschitz else nn.Conv3d(in_ch, in_ch // 2, kernel_size=1))
        else:
            if lipschitz:
                self.upsample = P.register_parametrization(
                    nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2), 'weight',
                    L.L2LipschitzConvTranspose3d(in_size, ks=2, stride=2, eps=power_eps, iterations=power_its, max_lc=max_lc))
            else:
                self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, act=act, slope=slope,
                                 lipschitz=lipschitz, max_lc=max_lc, power_its=power_its, power_eps=power_eps, in_size=in_size)

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


class BasicUnet3d(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        act = config.get('PARAMETERS', 'ACTIVATION')
        slope = config.getfloat('PARAMETERS', 'ACTIVATION_SLOPE')

        features = [features_start]
        for _ in range(1, 5):
            features.append(features[-1] * 2)
        features.append(features_start)

        self.unet = BasicUNet(spatial_dims=3, in_channels=input_channels, out_channels=num_classes,
                              act=("LeakyReLU", {"negative_slope": slope, "inplace": True}),
                              norm=("instance", {"affine": True}),
                              features=features)

    def forward(self, x):
        identity = x[:, 1:2, :, :, :].clone()
        x = self.unet(x)
        return x + identity, x


class ResUnet3d(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        slope = config.getfloat('PARAMETERS', 'ACTIVATION_SLOPE')
        num_res_units = config.getint('PARAMETERS', 'NUM_RES_UNITS')

        channels = [features_start]
        strides = []
        for _ in range(1, num_layers):
            channels.append(channels[-1] * 2)
            strides.append(2)

        self.unet = UNet(spatial_dims=3, in_channels=input_channels, out_channels=num_classes, channels=channels,
                         strides=strides, num_res_units=num_res_units,
                         act=("LeakyReLU", {"negative_slope": slope, "inplace": True}),
                         norm=("instance", {"affine": True}))

    def forward(self, x):
        identity = x[:, 1:2, :, :, :].clone()
        x = self.unet(x)
        return x + identity, x
