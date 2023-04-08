from monai.networks.nets import SwinUNETR
from torch import nn


class SwinUNet(nn.Module):
    def __init__(self):
        super().__init__()
