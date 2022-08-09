import torch
import torch.nn as nn
import sys
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, reduction, weight=None, size_average=True, alpha=0.7):
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1. - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        dice_bce = (1.0 - self.alpha) * bce + self.alpha * dice_loss

        return dice_bce


def loss_func_complete(mts, mtts):
    mse_loss = nn.MSELoss(reduction='sum')
    l1 = torch.tensor([1.0])
    N = len(mtts)
    loss = 0
    for i in range(N):
        loss += mse_loss(mts[i], mtts[i])
    loss = loss / N
    return (loss, l1, l1, l1)


def loss_func_three(mts, mtts, lambda_=1.0):
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
    l3 = (lambda_ / (N - 2)) * l3
    total_loss = l1 + l2 + l3
    return (total_loss, l1, l2, l3)


def loss_func_batch(mts, mtts, offsets, lambda_):
    # hubber_loss = nn.HuberLoss(reduction='sum', delta=1.35)
    # {m0, m1, m2, m3, m4, m4, m4}, {m0, m1, m2, m3, m4, m5, m6}
    # offsets = [2, 0]

    # {m0_b0, m0_b1}    mts[0]
    # {m1_b0, m1_b1}    mts[1]  k=1
    # {m2_b0, m2_b1}    mts[2]  k=2
    # {m3_b0, m3_b1}    mts[3]  k=3
    # {m4_b0, m4_b1}    mts[4]  k=4
    # {m4_b0, m5_b1}    mts[5]  k=5
    # {m4_b0, m6_b1}    mts[6]

    # {m4, m3, m2, m1, m0, m0, m0}, {m6, m5, m4, m3, m2, m1, m0}

    # {m4_b0, m6_b1}    mtts[0]     |   {m0_b0, m0_b1}    mtts[0]
    # {m3_b0, m5_b1}    mtts[1]     |   {m0_b0, m1_b1}    mtts[1]
    # {m2_b0, m4_b1}    mtts[2]     |   {m0_b0, m2_b1}    mtts[2]
    # {m1_b0, m3_b1}    mtts[3]     |   {m1_b0, m3_b1}    mtts[3]
    # {m0_b0, m2_b1}    mtts[4]     |   {m2_b0, m4_b1}    mtts[4]
    # {m0_b0, m1_b1}    mtts[5]     |   {m3_b0, m5_b1}    mtts[5]
    # {m0_b0, m0_b1}    mtts[6]     |   {m4_b0, m6_b1}    mtts[6]

    mse_loss = nn.MSELoss(reduction='sum')
    # BS = mts[0].shape[0]
    BS = mts.shape[1]
    # device = mts[0].device
    # dtype = mts[0].dtype
    # if BS == 1:
    #     return loss_func_three(mts, mtts, lambda_)[0]

    # compute l1
    # l1 = torch.zeros(BS, dtype=dtype, device=device)
    m0 = mts[0]
    m0tt = mtts[0]
    l1 = mse_loss(m0tt, m0)
    # for b in range(BS):
    #     idx = offsets[b]
    #     m0_b = m0[b, :, :, :, :].unsqueeze(0)
    #     m0tt_b = mtts[idx][b, :, :, :, :].unsqueeze(0)
    #     l1[b] = mse_loss(m0tt_b, m0_b)

    # compute l2
    # l2 = torch.zeros(BS, dtype=dtype, device=device)
    mk = mtts[-1]
    mkt = mts[-1]
    l2 = mse_loss(mkt, mk)
    # for b in range(BS):
    #     mk_b = mk[b, :, :, :, :].unsqueeze(0)
    #     mkt_b = mts[-offsets[b] - 1][b, :, :, :, :].unsqueeze(0)
    #     l2[b] = mse_loss(mkt_b, mk_b)

    # compute l3
    timesteps = mts.shape[0]
    # print(timesteps)
    l3 = 0
    for b in range(BS):
        mt = mts[1:timesteps - offsets[b] - 1, b, :, :, :, :]
        mtt = mtts[1 + offsets[b]:-1, b, :, :, :, :]
        l3 += mse_loss(mt, mtt) / mt.shape[0]

    # compute l3
    # timesteps = len(mts)
    # timesteps = mts.shape[0]
    # l3 = torch.zeros(BS, dtype=dtype, device=device)
    # ts_per_batch = torch.zeros(BS, dtype=dtype, device=device)
    # for k in range(1, timesteps - 1):
    #     mt = mts[k]
    #     for b in range(BS):
    #         idx = k + offsets[b]
    #         if idx < timesteps - 1:
    #             mt_b = mt[b, :, :, :, :].unsqueeze(0)
    #             # mtt_b = mtts[idx][b, :, :, :, :].unsqueeze(0)
    #             mtt_b = mtts[idx, b, :, :, :, :].unsqueeze(0)
    #             l3[b] += mse_loss(mt_b, mtt_b)
    #             ts_per_batch[b] += 1.0
    # l3 = l3 / ts_per_batch
    loss = l1 + l2 + (lambda_ * l3)
    return loss / BS


# class L2LossReduced(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, m0tt, mtk, m0, mk):
#         # total_loss = 0
#         # ctx.save_for_backward(mts, mtts)
#         # for k in range(len(mtts)):
#         #     loss = 0.5 * (mts[k] - mtts[k]).pow(2).sum()
#         #     total_loss += loss
#         R0 = m0tt - m0
#         RK = mtk - mk
#         ctx.save_for_backward(R0, RK)
#         return (0.5 * R0.pow(2).sum()) + (0.5 * RK.pow(2).sum())

#     @staticmethod
#     def backward(ctx, grad_out):
#         R0, RK = ctx.saved_tensors
#         return grad_out * R0, grad_out * RK, None, None


# class L2LossComplete(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mts, mtts):
#         total_loss = 0
#         # ctx.save_for_backward(mts, mtts)

#         R0 = mtts[0] - mts[0]
#         RK = mts[-1] - mtts[-1]
#         R02 = 0.5 * R0.pow(2).sum()
#         RK2 = 0.5 * RK.pow(2).sum()
#         # ctx.save_for_backward(mts, mtts)
#         ctx.mts = mts
#         ctx.mtts = mtts

#         total_loss = R02 + RK2
#         for j in range(1, len(mtts) - 1):
#             total_loss += 0.5 * (mts[j] - mtts[j]).pow(2).sum()
#         return total_loss

#     @staticmethod
#     def backward(ctx, grad_out):
#         # mts, mtts = ctx.saved_tensors
#         mts = ctx.mts
#         mtts = ctx.mtts
#         R0 = mtts[0] - mts[0]
#         RK = mts[-1] - mtts[-1]
#         RJ = 0
#         for j in range(1, len(mtts) - 1):
#             RJ += mts[j] - mtts[j]
#         dmt = RK + RJ
#         dmtt = R0 - RJ
#         return grad_out * dmt, grad_out * dmtt
