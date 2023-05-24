import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BasicUNet3d"]


class BasicUNet3d(nn.Module):
    def __init__(self, config, logger):
        super(BasicUNet3d, self).__init__()

        n_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        n_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        bilinear = config.getboolean('PARAMETERS', 'TRILINEAR')
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')
        self.out_layer = config.get('PARAMETERS', 'OUT_LAYER')

        feats = [features_start]
        for i in range(4):
            feats.append(feats[-1] * 2)

        self.inc = DoubleConv(n_channels, feats[0])
        self.down1 = Down(feats[0], feats[1])
        self.down2 = Down(feats[1], feats[2])
        self.down3 = Down(feats[2], feats[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(feats[3], feats[4] // factor)
        self.up1 = Up(feats[4], feats[3] // factor, bilinear)
        self.up2 = Up(feats[3], feats[2] // factor, bilinear)
        self.up3 = Up(feats[2], feats[1] // factor, bilinear)
        self.up4 = Up(feats[1], feats[0], bilinear)
        self.outc = OutConv(feats[0], n_classes)

    def forward(self, x):
        if self.residual:
            identity = x[:, 1:]

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        xu1 = x

        x = self.up2(x, x3)
        r1 = self.up2(xu1, x) + x

        x = self.up3(x, x2)
        r2 = self.up3(r1, x) + x

        x = self.up4(x, x1)
        r3 = self.up4(r2, x) + x

        # logits = self.outc(x)
        logits = self.outc(r3)

        if self.residual:
            return logits + identity, logits
        else:
            logits_c = logits
            logits = torch.sigmoid(logits) if self.out_layer == 'sigmoid' else logits
            return logits, logits_c


class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
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
                        diffZ // 2, diffZ - diffZ // 2, ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
