import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class TransformerUNet(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        num_layers = config.getint('PARAMETERS', 'NUM_LAYERS')
        num_classes = config.getint('PARAMETERS', 'NUM_CLASSES')
        input_channels = config.getint('PARAMETERS', 'INPUT_CHANNELS')
        features_start = config.getint('PARAMETERS', 'FEATURES_START')
        slope = config.getfloat('PARAMETERS', 'ACTIVATION_SLOPE')
        num_res_units = config.getint('PARAMETERS', 'NUM_RES_UNITS')
        self.residual = config.getboolean('PARAMETERS', 'RESIDUAL')
        self.out_layer = config.get('PARAMETERS', 'OUT_LAYER')

        self.net = UNETR(in_channels=input_channels,out_channels=num_classes)