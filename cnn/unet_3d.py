"""Adapted from <https://github.com/annikabrundyn> and <https://github.com/akshaykvnit>"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, config, logger):
        num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        dropout = config.getboolean('PARAMETERS', 'DROPOUT')
        trilinear = config.getboolean('PARAMETERS', 'TRILINEAR')
        dp = config.getfloat('PARAMETERS', 'DROPOUT_PROB')
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
                \nDropout: {dropout}\
                \nDropout prob: {dp}\
                \nTrilinear interp: {trilinear}\
                \nPadding: {padding}\
                \nKernel size: {kernel_size}\
                \nNorm: {self.norm}\
                \nResidual net: {self.residual}")
            logger.info("==================================")

        if num_layers < 1:
            raise ValueError(
                f"Num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers
        layers = [DoubleConv3D(input_channels, features_start,
                               dropout, dp, kernel_size, padding, self.norm, self.act, self.slope)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down3D(feats, feats * 2, dropout,
                          dp, kernel_size, padding, self.norm, self.act, self.slope))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up3D(feats, feats // 2, trilinear,
                          dropout, dp, kernel_size, padding, self.norm, self.act, self.slope))
            feats //= 2

        layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        identity_mask = x[:, 1:2, :, :, :] if self.residual else None

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        return (self.layers[-1](xi[-1])) + identity_mask if self.residual else (self.layers[-1](xi[-1]))


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout=False, dp=0.5, kernel_size=(3, 3, 3), padding=1, norm='IN', act='relu', slope=0.2):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch) if norm == 'IN' else nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch) if norm == 'IN' else nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(negative_slope=slope, inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout3d(dp))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False, dp=0.5, kernel_size=(3, 3, 3), padding=1, norm='IN', act='relu', slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_ch, out_ch, dropout, dp, kernel_size, padding, norm, act, slope)
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """Upsampling (by either trilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv3D."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = False, dropout: bool = False, dp=0.5, kernel_size=(3, 3),
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

        self.conv = DoubleConv3D(in_ch, out_ch, dropout, dp, kernel_size=kernel_size, padding=padding, norm=norm, act=act, slope=slope)

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
