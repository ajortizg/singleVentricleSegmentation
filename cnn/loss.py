import torch
from torch import nn


class CustomLoss(nn.Module):
    def __init__(self, lambda_v, penalization, gamma_v):
        super().__init__()
        self.penalization = penalization
        self.lambda_v = lambda_v
        self.gamma_v = gamma_v

    def forward(self, mts, mtts, mhs, mhhs, offsets, batch_indices):
        batch_size = mts.shape[1]

        # Supervised term
        m0 = mts[0, batch_indices]
        m0tt = mtts[offsets[batch_indices], batch_indices]
        mk = mtts[-1, batch_indices]
        mkt = mts[-offsets[batch_indices] - 1, batch_indices]
        LS = ((m0 - m0tt).pow(2).sum() + (mk - mkt).pow(2).sum()) * (self.lambda_v / batch_size)

        # Self-supervised and penalization term
        LU = 0
        kl = mts.shape[0] - offsets
        ts_tildes = mts.shape[0]
        LP = torch.tensor([0.0], dtype=LS.dtype, device=LS.device)
        ts_hats = mhs.shape[0] if self.penalization else 0

        for b in range(batch_size):
            # Self-supervised term
            mt = mts[1:ts_tildes - offsets[b] - 1, b]
            mtt = mtts[1 + offsets[b]:-1, b]
            LU += (mt - mtt).pow(2).sum() / kl[b]

            # Penalization term
            if self.penalization:
                mh = mhs[0:ts_hats - offsets[b], b]
                mhh = mhhs[0:ts_hats - offsets[b], b]
                LP += (mh.norm().sum() + mhh.norm().sum()) / kl[b]
                # LP += (mh.pow(2).sum() + mhh.pow(2).sum()) / kl[b]

        LU *= (self.lambda_v / batch_size)
        LP *= (self.gamma_v / batch_size)
        LT = LS + LU + LP
        return LT, LS, LU, LP
