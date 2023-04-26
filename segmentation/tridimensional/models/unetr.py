from configparser import ConfigParser

import torch
import torch.nn as nn
from monai.networks import nets


class UNetr(nn.Module):
    def __init__(self, cfg: ConfigParser):
        super().__init__()
        num_classes = cfg.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = cfg.getint('PARAMETERS', 'INPUT_CHANNELS')
        img_size = cfg.getint('PARAMETERS', 'IMG_SIZE')
        features_start = cfg.getint('PARAMETERS', 'FEATURES_START')
        hidden_size = cfg.getint('PARAMETERS', 'HIDDEN_SIZE')
        mlp_dim = cfg.getint('PARAMETERS', 'MLP_DIM')
        num_heads = cfg.getint('PARAMETERS', 'NUM_HEADS')

        self.net = nets.UNETR(in_channels=input_channels,
                              out_channels=num_classes,
                              img_size=img_size,
                              hidden_size=hidden_size,
                              mlp_dim=mlp_dim,
                              num_heads=num_heads,
                              feature_size=features_start,
                              spatial_dims=3)

    def forward(self, x):
        return self.net(x)
