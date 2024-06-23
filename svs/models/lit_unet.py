from typing import Tuple
import lightning as L
import torch

from svs.models.unet import UNet


class LitUNet(L.LightningModule):
    def __init__(
        self,
        epochs,
        lr: float = 1e-4,
        betas: Tuple[float] = (0.9, 0.999),
        gamma: float = 1e-1,
        weight_decay: float = 1e-4,
        model_kwargs: dict = {},
        **kwargs: torch.Any
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model = UNet(**model_kwargs)
