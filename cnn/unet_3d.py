"""Adapted from <https://github.com/annikabrundyn> and <https://github.com/akshaykvnit>"""
import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
from monai.networks.nets.unet import UNet


class UNet3D(nn.Module):
    def __init__(self, config, logger):
        num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        trilinear = config.getboolean('PARAMETERS', 'TRILINEAR')
        padding = config.getint('PARAMETERS', 'PADDING')
        verbose = config.getboolean('DEBUG', 'VERBOSE')
        kstr = config.get('PARAMETERS', 'KERNEL_SIZE')
        kernel_size = tuple(map(int, kstr.split(',')))
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')
        self.norm = config.get('PARAMETERS', 'NORM_LAYER')
        self.act = config.get('PARAMETERS', 'ACTIVATION')
        self.slope = config.getfloat('PARAMETERS', 'ACTIVATION_SLOPE')

        if verbose:
            logger.info("\n================CNN===============")
            logger.info(f"Layers: {num_layers}\
                \nClasses: {num_classes}\
                \nInput channels: {input_channels}\
                \nFeatures start: {features_start}\
                \nTrilinear interp: {trilinear}\
                \nPadding: {padding}\
                \nKernel size: {kernel_size}\
                \nNorm: {self.norm}\
                \nAct: {self.act}\
                \nAct slope: {self.slope}\
                \nResidual net: {self.residual}")
            logger.info("==================================")

        if num_layers < 1:
            raise ValueError(
                f"Num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers
        layers = [DoubleConv3D(input_channels, features_start, kernel_size, padding, self.norm, self.act, self.slope)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down3D(feats, feats * 2, kernel_size, padding, self.norm, self.act, self.slope))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up3D(feats, feats // 2, trilinear, kernel_size, padding, self.norm, self.act, self.slope))
            feats //= 2

        layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity_mask = x[:, 1:2, :, :, :] if self.residual else None

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        if self.residual:
            return self.sigmoid(self.layers[-1](xi[-1]) + identity_mask)
        else:
            return self.sigmoid(self.layers[-1](xi[-1]))


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, norm='IN', act='relu', slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch) if norm == 'IN' else nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch) if norm == 'IN' else nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3, 3), padding=1, norm='IN', act='relu', slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_ch, out_ch, kernel_size, padding, norm, act, slope)
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """Upsampling (by either trilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv3D."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = False, kernel_size=(3, 3),
                 padding=1, norm='IN', act='relu', slope=0.2):
        super().__init__()
        self.upsample = None
        if trilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                nn.Conv3d(in_ch, in_ch // 2, kernel_size=1)
            )
        else:
            self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_ch, out_ch, kernel_size=kernel_size, padding=padding, norm=norm, act=act, slope=slope)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_d = x2.shape[2] - x1.shape[2]
        diff_h = x2.shape[3] - x1.shape[3]
        diff_w = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1,
                   [diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2]
                   )
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


""" Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet """


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.n_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        self.n_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        self.trilinear = config.getboolean('PARAMETERS', 'TRILINEAR')
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')

        self.inc = DoubleConv(self.n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if self.trilinear else 1
        # self.down4 = Down(64, 128 // factor)
        self.up1 = Up(64, 32 // factor, self.trilinear)
        self.up2 = Up(32, 16 // factor, self.trilinear)
        self.up3 = Up(16, 8 // factor, self.trilinear)
        # self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, self.n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity_mask = x[:, 1:2, :, :, :] if self.residual else None

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        logits = self.outc(x)

        if self.residual:
            return self.sigmoid(logits + identity_mask)
        else:
            return self.sigmoid(logits)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
