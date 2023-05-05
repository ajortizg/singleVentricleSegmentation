from configparser import ConfigParser

import torch
import torch.nn as nn
from monai.networks import nets


class UNetr(nn.Module):
    def __init__(self, n_classes: int, cfg: ConfigParser):
        super().__init__()
        params = cfg['PARAMETERS']
        n_classes = n_classes + 1 if n_classes > 1 else n_classes
        in_channels = n_classes + 1
        img_size = params.getint('img_size')
        features_start = params.getint('features_start')
        hidden_size = params.getint('hidden_size')
        mlp_dim = params.getint('mlp_dim')
        num_heads = params.getint('num_heads')

        self.net = nets.UNETR(in_channels=in_channels,
                              out_channels=n_classes,
                              img_size=img_size,
                              hidden_size=hidden_size,
                              mlp_dim=mlp_dim,
                              num_heads=num_heads,
                              feature_size=features_start,
                              spatial_dims=3)

    def forward(self, x):
        identity = x[:, 1:]
        output = self.net(x)
        return output + identity, output
