from typing import Tuple, Dict, Any
import lightning as pl
import torch

from svs.models.unet import UNet


class LitUNet(pl.LightningModule):
    def __init__(
        self,
        epochs,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        model_kwargs: Dict[str, Any] = {},
        **kwargs: torch.Any
    ):
        """
        Initializes the LitUNet model.

        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
            betas (Tuple[float, float]): Betas for the Adam optimizer.
            weight_decay (float): Weight decay for the optimizer.
            model_kwargs (Dict[str, Any]): Arguments for the UNet model.
            **kwargs (Any): Additional arguments for the LightningModule.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model = UNet(**model_kwargs)

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def configure_optimizers(self, lr: float = None):
        lr = lr or self.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=self.betas, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=lr / 100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        pass

    def propagation(self):
        pass
