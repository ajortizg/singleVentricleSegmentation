import torch
from torch import nn


class CustomLoss(nn.Module):
    def __init__(self, lambda_ss, gamma_p):
        super().__init__()
        self.lambda_ss = lambda_ss
        self.gamma_p = gamma_p

    def forward(self, mts, mtts, mhs, mhhs, offsets, batch_idxs):
        bs = mts.shape[1]

        # Supervised term
        mi_true = mts[0, batch_idxs]
        mi_pred = mtts[offsets[batch_idxs], batch_idxs]
        mf_true = mtts[-1, batch_idxs]
        mf_pred = mts[-offsets[batch_idxs] - 1, batch_idxs]
        LS = ((mi_true - mi_pred).pow(2).sum() + (mf_true - mf_pred).pow(2).sum()) / bs

        # Self-supervised and penalization term
        LU = 0
        kl = mts.shape[0] - offsets
        ts_tildes = mts.shape[0]
        LP = 0
        ts_hats = mhs.shape[0]

        for b in range(bs):
            # Self-supervised term
            mt = mts[1:ts_tildes - offsets[b] - 1, b]
            mtt = mtts[1 + offsets[b]:-1, b]
            LU += (mt - mtt).pow(2).sum() / kl[b]

            # Penalization term
            mh = mhs[0:ts_hats - offsets[b], b]
            mhh = mhhs[0:ts_hats - offsets[b], b]
            # LP += (mh.norm().sum() + mhh.norm().sum()) / kl[b]
            LP += (mh.pow(2).sum() + mhh.pow(2).sum()) / kl[b]

        LU *= (self.lambda_ss / bs)
        LP *= (self.gamma_p / bs)
        LT = LS + LU + LP
        return LT, LS, LU, LP
