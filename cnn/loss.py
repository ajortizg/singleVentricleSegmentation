import torch
import torch.nn as nn


def loss_func_complete(mts, mtts):
    mse_loss = nn.MSELoss(reduction='sum')
    m0 = mts[0]
    m0tt = mtts[0]
    mk = mtts[-1]
    mkt = mts[-1]

    l1 = mse_loss(m0tt, m0)
    l2 = mse_loss(mkt, mk)
    l3 = 0
    N = len(mtts)
    for i in range(1, N - 1):
        l3 += mse_loss(mts[i], mtts[i])
    l3 = l3 / (N - 2)
    total_loss = l1 + l2 + l3
    return (total_loss, l1, l2, l3)
    # l1 = torch.tensor([1.0])
    # N = len(mtts)
    # loss = 0
    # for i in range(N):
    #     loss += mse_loss(mts[i], mtts[i])
    # loss = loss / N
    # return (loss, l1, l1, l1)


class L2LossReduced(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m0tt, mtk, m0, mk):
        # total_loss = 0
        # ctx.save_for_backward(mts, mtts)
        # for k in range(len(mtts)):
        #     loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
        #     total_loss += loss
        R0 = m0tt - m0
        RK = mtk - mk
        ctx.save_for_backward(R0, RK)
        return (0.5 * R0.pow(2).sum()) + (0.5 * RK.pow(2).sum())

    @staticmethod
    def backward(ctx, grad_out):
        R0, RK = ctx.saved_tensors
        return grad_out * R0, grad_out * RK, None, None


class L2LossComplete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mts, mtts):
        total_loss = 0
        # ctx.save_for_backward(mts, mtts)

        R0 = mtts[0] - mts[0]
        RK = mts[-1] - mtts[-1]
        R02 = 0.5 * R0.pow(2).sum()
        RK2 = 0.5 * RK.pow(2).sum()
        # ctx.save_for_backward(mts, mtts)
        ctx.mts = mts
        ctx.mtts = mtts

        total_loss = R02 + RK2
        for j in range(1, len(mtts) - 1):
            total_loss += 0.5 * (mts[j] - mtts[j]).pow(2).sum()
        return total_loss

    @staticmethod
    def backward(ctx, grad_out):
        # mts, mtts = ctx.saved_tensors
        mts = ctx.mts
        mtts = ctx.mtts
        R0 = mtts[0] - mts[0]
        RK = mts[-1] - mtts[-1]
        RJ = 0
        for j in range(1, len(mtts) - 1):
            RJ += mts[j] - mtts[j]
        dmt = RK + RJ
        dmtt = R0 - RJ
        return grad_out * dmt, grad_out * dmtt
