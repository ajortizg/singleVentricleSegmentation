"""Adapted from <https://github.com/annikabrundyn> and <https://github.com/akshaykvnit>"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class UNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        input_channels: int,
        features_start: int,
        trilinear: bool = False,
        padding: int = 1,
        kernel_size: Tuple[int, int, int] = (3, 3, 3)
    ):
        super(UNet, self).__init__()

        if num_layers < 1:
            raise ValueError(f"Num_layers = {num_layers}, expected: num_layers > 0")

        self.num_layers = num_layers
        layers = [DoubleConv3d(input_channels, features_start, kernel_size, padding)]
        feats = features_start

        for _ in range(num_layers - 1):
            layers.append(Down3d(feats, feats * 2, kernel_size, padding))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up3d(feats, feats // 2, trilinear, kernel_size, padding))
            feats //= 2

        layers.append(nn.Conv3d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: int = 1
    ):
        super(DoubleConv3d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: int = 1
    ):
        super(Down3d, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3d(in_ch, out_ch, kernel_size, padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up3d(nn.Module):
    """
    Upsampling (by either trilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv3D.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        trilinear: bool = False,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        padding: int = 1
    ):
        super(Up3d, self).__init__()
        self.upsample = None
        if trilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="trilinear",
                    align_corners=True
                ),
                nn.Conv3d(in_ch, in_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
