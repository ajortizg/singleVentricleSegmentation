import torch
# import numpy as np

eps = 1e-10


def dice(pred: torch.Tensor, target: torch.Tensor):
    """
    Dice coefficient
    y_true and y_pred are tensors with shape [BS, CH, NZ, NY, NX]
    Returns the mean along the batch dimension
    """
    # # y_true = torch.where(y_true > 0.5, 1.0, 0.0)
    # # y_pred = torch.where(y_pred > 0.5, 1.0, 0.0)
    # intersection = torch.sum(y_true * y_pred, dim=axis)
    # summation = torch.sum(y_true, dim=axis) + torch.sum(y_pred, dim=axis)
    # return torch.mean((2.0 * intersection + eps) / (summation + eps))

    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# def dice_2(y_true: torch.Tensor, y_pred: torch.Tensor, axis: list = [1, 2, 3, 4]):
#     x = torch.where(y_true > 0.5, 1.0, 0.0)
#     y = torch.where(y_pred > 0.5, 1.0, 0.0)

#     x = torch.atleast_1d(x).to(torch.bool)
#     y = torch.atleast_1d(y).to(torch.bool)

#     intersection = torch.count_nonzero(x & y)

#     size_i1 = torch.count_nonzero(x)
#     size_i2 = torch.count_nonzero(y)

#     try:
#         dc = 2. * intersection / float(size_i1 + size_i2)
#     except ZeroDivisionError:
#         dc = 0.0

#     return dc

# def dc(x, y):
#     r"""
#     Dice coefficient

#     Computes the Dice coefficient (also known as Sorensen index) between the binary
#     objects in two images.

#     The metric is defined as

#     .. math::

#         DC=\frac{2|A\cap B|}{|A|+|B|}

#     , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

#     Parameters
#     ----------
#     result : array_like
#         Input data containing objects. Can be any type but will be converted
#         into binary: background where 0, object everywhere else.
#     reference : array_like
#         Input data containing objects. Can be any type but will be converted
#         into binary: background where 0, object everywhere else.

#     Returns
#     -------
#     dc : float
#         The Dice coefficient between the object(s) in ```result``` and the
#         object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

#     Notes
#     -----
#     This is a real metric. The binary images can therefore be supplied in any order.
#     """
#     x = np.atleast_1d(x.astype(np.bool))
#     y = np.atleast_1d(y.astype(np.bool))

#     intersection = np.count_nonzero(x & y)

#     size_i1 = np.count_nonzero(x)
#     size_i2 = np.count_nonzero(y)

#     try:
#         dc = 2. * intersection / float(size_i1 + size_i2)
#     except ZeroDivisionError:
#         dc = 0.0

#     return dc
