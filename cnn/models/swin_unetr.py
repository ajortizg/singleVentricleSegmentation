from monai.networks.nets import SwinUNETR
from torch import nn
import torch

__all__ = ["SwinUNetr"]

class SwinUNetr(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()
        param_cfg = cfg['PARAMETERS']
        num_classes = param_cfg.getint('NUM_CLASSES')
        input_channels = param_cfg.getint('INPUT_CHANNELS')
        img_size = param_cfg.getint('IMG_SIZE')
        features_start = param_cfg.getint('FEATURES_START')
        self.residual = param_cfg.getboolean('RESIDUAL')
        self.out_layer = param_cfg.get('OUT_LAYER')

        self.net = SwinUNETR(img_size=(img_size, img_size, img_size),
                             in_channels=input_channels,
                             out_channels=num_classes,
                             depths=(1, 1, 2, 2),
                             num_heads=(3, 6, 12, 12),
                             feature_size=12,
                             norm_name='instance',
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

