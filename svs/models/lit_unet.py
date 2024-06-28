from typing import Tuple, Dict, Any, Iterable
import lightning as pl
import torch
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance


from svs.models.unet import UNet
from svs.modules.flow.warping import Warp
from svs.modules.loss import PropagationLoss
from svs.utils.constants import *


class LitUNet(pl.LightningModule):
    def __init__(
        self,
        epochs,
        lambda_u: float,
        gamma_p: float,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        model_kwargs: Dict[str, Any] = {},
        warp_kwargs: Dict[str, Any] = {},
        ** kwargs: torch.Any
    ):
        """
        Args:
            epochs (int): Number of training epochs.
            lambda_u (float): Weight for the unsupervised loss term.
            gamma_p (float): Weight for the penalty term.
            lr (float): Learning rate for the optimizer.
            betas (Tuple[float, float]): Betas for the Adam optimizer.
            weight_decay (float): Weight decay for the optimizer.
            model_kwargs (Dict[str, Any]): Arguments for the UNet model.
            warp_kwargs (Dict[str, Any]): Arguments for the Warp module.
            **kwargs (Any): Additional arguments for the LightningModule.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model = UNet(**model_kwargs)
        self.warp = Warp(**warp_kwargs)
        self.loss_fn = PropagationLoss(lambda_u, gamma_p)

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def configure_optimizers(self, lr: float = None):
        lr = lr or self.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=self.betas, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=lr / 100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, batch: Dict[str, Any]) -> Iterable[torch.Tensor]:
        img, mi, mf, fflow, bflow, ftimes, btimes = self.extract_batch_data(batch)

        bs, ts, nz, ny, nx, _ = fflow.shape
        dtype = mi.dtype
        device = mi.device
        batch_indices = torch.arange(bs)

        # Initialize tensors for forward and backward propagation
        # mts -> forward
        # mtts -> backward
        mts = torch.empty(size=(bs, ts + 1, 1, nz, ny, nx), dtype=dtype, device=device)
        mts[:, 0] = mi
        mtts = torch.empty_like(mts)
        mtts[:, -1] = mf

        # CNN outputs without residual connection, used in the loss computation
        mhs = torch.empty(size=(bs, ts, 1, nz, ny, nx), dtype=dtype, device=device)
        mhhs = torch.empty_like(mhs)

        for i in range(ts):
            mts[:, i + 1], mhs[:, i] = self.propagation_step(mts[:, i], img, fflow[:, i], ftimes, i, batch_indices)
            mtts[:, ts - i - 1], mhhs[:, ts - i - 1] = self.propagation_step(mtts[:, ts - i], img, bflow[:, i], btimes, i, batch_indices)

        return mts, mtts, mhs, mhhs

    def extract_batch_data(self, batch: Dict[str, Any]) -> Iterable[torch.Tensor]:
        img = batch[IMAGE_KEY]          # (B, T, C, Z, Y, X)
        mi = batch[MI_KEY]              # (B, C, Z, Y, X)
        mf = batch[MF_KEY]              # (B, C, Z, Y, X)
        fflow = batch[FWD_FLOW_KEY]     # (B, T, Z, Y, X, 3)
        bflow = batch[BWD_FLOW_KEY]     # (B, T, Z, Y, X, 3)
        fts = batch[FWD_TS_KEY]
        bts = batch[BWD_TS_KEY]
        return img, mi, mf, fflow, bflow, fts, bts

    def propagation_step(
        self,
        mask: torch.Tensor,
        img: torch.Tensor,
        flow: torch.Tensor,
        times: torch.Tensor,
        i: int,
        batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a single propagation step (forward or backward)."""
        warped_mask = self.warp(mask, flow)

        # TODO: batch_indices is needed in fts?
        x = torch.cat((img[batch_indices, times[batch_indices, i + 1]], warped_mask), dim=1)
        next_mask, mh = self.model(x)

        return next_mask, mh

    def forward_and_loss(self, batch, batch_idx) -> Tuple[Tuple[torch.Tensor]]:
        y = self(batch)

        offsets = torch.tensor(batch[OFFSET_KEY]).to(torch.long)
        loss = self.loss_fn(y[0], y[1], y[2], y[3], offsets)

        return loss, y

    def training_step(self, batch, batch_idx):
        loss, y = self.forward_and_loss(batch, batch_idx)

        # Log metrics for each training_step
        for i in range(len(loss)):
            self.log("train_loss_{i}", loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log_dict()

        return loss[0]

    def validation_step(self, batch, batch_idx):
        loss, y = self.forward_and_loss(batch, batch_idx)

        # # Plot loss in tensorboard
        # for i, (train, val) in enumerate(
        #     zip(self.mean_epoch_stat['train_loss'][-1],
        #         self.mean_epoch_stat['val_loss'][-1])):
        #     self.writer.add_scalars(f'loss/l{i}', {'train': train, 'val': val}, e)
