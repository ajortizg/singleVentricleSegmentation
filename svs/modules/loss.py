import torch
from torch import nn
from typing import Dict

from svs.utils.constants import *


class PropagationLoss(nn.Module):
    def __init__(self, lambda_u: float, gamma_p: float):
        """
        Args:
            lambda_unsup (float): Weight for the unsupervised loss term.
            gamma_penalty (float): Weight for the penalty term.
        """
        super().__init__()

        self.lambda_u = lambda_u
        self.gamma_p = gamma_p
        self.mse_fn = nn.MSELoss(reduction="sum")

    def forward(self, y: tuple[torch.Tensor], offsets: torch.Tensor) -> Dict[str, torch.Tensor]:
        mts, mtts, mhs, mhhs = y
        bs = mts.shape[0]
        batch_indices = torch.arange(bs)

        # Supervised term for known labels
        mi = mts[batch_indices, 0]
        mitt = mtts[batch_indices, offsets[batch_indices]]
        sup_term_1 = self.mse_fn(mitt, mi)

        mf = mtts[batch_indices, -1]
        mft = mts[batch_indices, -offsets[batch_indices] - 1]
        sup_term_2 = self.mse_fn(mft, mf)

        # Unsupervised term for interior labels (no ground truth)
        kl = mts.shape[1] - offsets
        times_tildes = mts.shape[1]
        times_hats = mhs.shape[1]

        unsup_term = torch.as_tensor(0.0, device=mts.device)
        pen_term = torch.as_tensor(0.0, device=mts.device)  # Penalization term

        for b in range(bs):
            # Self-supervised term
            mt = mts[b, 1:times_tildes - offsets[b] - 1]
            mtt = mtts[b, 1 + offsets[b]:-1]
            unsup_term += self.mse_fn(mt, mtt) / kl[b]

            # Penalization term
            mh = mhs[b, 0:times_hats - offsets[b]]
            mhh = mhhs[b, 0:times_hats - offsets[b]]
            pen_term += (mh.pow(2).sum() + mhh.pow(2).sum()) / kl[b]

        sup_term = (sup_term_1 + sup_term_2) / bs
        unsup_term *= (self.lambda_u / bs)
        pen_term *= (self.gamma_p / bs)
        loss = sup_term + unsup_term + pen_term

        return {
            TOTAL_LOSS_KEY: loss,
            SUPERVISED_LOSS_KEY: sup_term,
            UNSUPERVISED_LOSS_KEY: unsup_term,
            PENALIZATION_LOSS_KEY: pen_term
        }
