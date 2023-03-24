import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class TransformerUNet(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        img_size = config.getint('PARAMETERS', 'img_size')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
    
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')
        self.out_layer = config.get('PARAMETERS', 'OUT_LAYER')

        self.net = UNETR(in_channels=input_channels,
                         out_channels=num_classes,
                         img_size=img_size,
                         spatial_dims=3)