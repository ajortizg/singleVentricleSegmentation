from typing import Tuple, Dict, Any, Iterable
import lightning as pl
import torch
import torchmetrics

from svs.models.unet import UNet
from svs.modules.flow.warping import Warp
from svs.modules.loss import PropagationLoss
from svs.utils.constants import *
from svs.modules.metrics import Dice, Hausdorff


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

        self.setup_metrics()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas

    def setup_metrics(self):
        _metrics = torchmetrics.MetricCollection({
            "dice": Dice(),
            "hsdf": Hausdorff()
        })

        self.metrics = torch.nn.ModuleDict(dict(
            trn=torch.nn.ModuleList([_metrics.clone(postfix=f"_trn/{direction}") for direction in ["b", "f"]]),
            val=torch.nn.ModuleList([_metrics.clone(postfix=f"_val/{direction}") for direction in ["b", "f"]])
        ))

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
        ftimes = batch[FWD_TS_KEY]
        btimes = batch[BWD_TS_KEY]
        return img, mi, mf, fflow, bflow, ftimes, btimes

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

    def forward_and_loss(self, batch, batch_idx) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        y = self(batch)
        loss = self.loss_fn(y, batch[OFFSET_KEY])
        return loss, y

    def training_step(self, batch, batch_idx):
        loss, y = self.forward_and_loss(batch, batch_idx)

        # Log metrics for each training_step
        self.log_dict(
            {f"loss_trn/{k}": v for k, v in loss.items()},
            prog_bar=False,
            on_epoch=True,
            logger=True,
            batch_size=len(batch[OFFSET_KEY])
        )
        self.update_metrics("trn", y, batch[OFFSET_KEY])

        return loss[TOTAL_LOSS_KEY]

    def on_train_epoch_end(self):
        for i in range(2):
            self.log_dict(self.metrics["trn"][i].compute())
            self.metrics["trn"][i].reset()

    def validation_step(self, batch, batch_idx):
        loss, y = self.forward_and_loss(batch, batch_idx)

        # Log metrics for each validation step
        self.log_dict(
            {f"loss_val/{k}": v for k, v in loss.items()},
            prog_bar=False,
            on_epoch=True,
            logger=True,
            batch_size=len(batch[OFFSET_KEY])
        )
        self.update_metrics("val", y, batch[OFFSET_KEY])

        return loss[TOTAL_LOSS_KEY]

    def on_validation_epoch_end(self):
        for i in range(2):
            self.log_dict(self.metrics["val"][i].compute())
            self.metrics["val"][i].reset()

    @torch.no_grad()
    def update_metrics(self, key: str, y: Tuple[torch.Tensor], offsets: torch.Tensor):
        mts, mtts, *_ = y
        batch_indices = torch.arange(mts.shape[0])

        mi = mts[batch_indices, 0]
        mitt = mtts[batch_indices, offsets[batch_indices]]
        mf = mtts[batch_indices, -1]
        mft = mts[batch_indices, -offsets[batch_indices] - 1]

        for i, (y, y_hat) in enumerate(zip([mi, mf], [mitt, mft])):
            y_hat = torch.where(y_hat > 0.5, 1.0, 0.0)
            self.metrics[key][i].update(y_hat, y)
