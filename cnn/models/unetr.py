import torch
import torch.nn as nn
from monai.networks.nets import UNETR


__all__ = ["TransformerUNet"]


class TransformerUNet(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        img_size = config.getint('PARAMETERS', 'IMG_SIZE')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        hidden_size = config.getint('PARAMETERS', 'HIDDEN_SIZE')
        mlp_dim = config.getint('PARAMETERS', 'MLP_DIM')
        num_heads = config.getint('PARAMETERS', 'NUM_HEADS')
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')
        self.out_layer = config.get('PARAMETERS', 'OUT_LAYER')

        self.net = UNETR(in_channels=input_channels,
                         out_channels=num_classes,
                         img_size=img_size,
                         hidden_size=hidden_size,
                         mlp_dim=mlp_dim,
                         num_heads=num_heads,
                         feature_size=features_start,
                         spatial_dims=3)

    def forward(self, x):
        if self.residual:
            identity = x[:, 1:]

        output = self.net(x)

        if self.residual:
            return output + identity, output
        else:
            output_c = output
            output = torch.sigmoid(output) if self.out_layer == 'sigmoid' else output
            return output, output_c
