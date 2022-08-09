import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class L2LipschitzConv3d(nn.Module):
    def __init__(self, in_size, ks, padding=0, stride=1, eps=1e-8, iterations=1, max_lc=1.0):
        super(L2LipschitzConv3d, self).__init__()
        self.in_size = in_size
        self.padding = padding
        self.stride = stride
        self.ks = ks
        self.eps = eps
        self.iterations = iterations
        self.max_lc = max_lc

    def forward(self, w):
        norm = spectral_norm_conv3d_2(w, self.in_size, self.ks, self.padding, self.stride, self.eps, self.iterations)
        return w * (1.0 / max(1.0, norm / self.max_lc))


class L2LipschitzConvTranspose3d(nn.Module):
    def __init__(self, in_size, ks, padding=0, stride=1, eps=1e-8, iterations=1, max_lc=1.0):
        super(L2LipschitzConvTranspose3d, self).__init__()
        self.in_size = in_size
        self.padding = padding
        self.stride = stride
        self.ks = ks
        self.eps = eps
        self.iterations = iterations
        self.max_lc = max_lc

    def forward(self, w):
        norm = spectral_norm_conv_transpose3d_2(w, self.in_size, self.ks, self.padding, self.stride, self.eps, self.iterations)
        return w * (1.0 / max(1.0, norm / self.max_lc))


def lipschitz_bn(bn_layer):
    return max(abs(bn_layer.weight / torch.sqrt(bn_layer.running_var + bn_layer.eps)))


def spectral_norm_conv3d_2(w, input_size, ks, padding, stride, eps, max_iter):
    input_shape = (1, w.shape[1], input_size, input_size, input_size)
    x = torch.rand(input_shape, device=w.device)

    for i in range(0, max_iter):
        x_p = F.conv3d(x, w, stride=stride, padding=padding)
        x = F.conv_transpose3d(x_p, w, stride=stride, padding=padding)
    Wx = F.conv3d(x, w, stride=stride, padding=padding)
    norm = torch.sqrt(torch.sum(torch.pow(Wx, 2.0)) / torch.sum(torch.pow(x, 2.0)))
    return norm


def spectral_norm_conv_transpose3d_2(w, input_size, ks, padding, stride, eps, max_iter):
    input_shape = (1, w.shape[0], input_size, input_size, input_size)
    x = torch.rand(input_shape, device=w.device)

    for i in range(0, max_iter):
        x_p = F.conv_transpose3d(x, w, stride=stride, padding=padding)
        x = F.conv3d(x_p, w, stride=stride, padding=padding)

    Wx = F.conv_transpose3d(x, w, stride=stride, padding=padding)
    norm = torch.sqrt(torch.sum(torch.pow(Wx, 2.0)) / torch.sum(torch.pow(x, 2.0)))
    return norm


@torch.enable_grad()
def _norm_gradient_sq(linear_fn, v):
    v = Variable(v, requires_grad=True)
    loss = torch.norm(linear_fn(v))**2
    loss.backward()
    return v.grad.data


def spectral_norm_conv_transpose3d(w, input_size, ks, padding, stride, eps, max_iter):
    tc = nn.ConvTranspose3d(w.shape[0], w.shape[1], kernel_size=ks, stride=stride, padding=padding, bias=False)
    tc.weight.data = w.data
    input_shape = (1, w.shape[0], input_size, input_size, input_size)

    v = torch.randn(input_shape, device=w.device)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(tc, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    u = tc(Variable(v))  # unormalized left singular vector
    singular_value = torch.norm(u).item()
    return singular_value


def spectral_norm_conv3d(w, input_size, ks, padding, stride, eps, max_iter):
    conv = torch.nn.Conv3d(w.shape[0], w.shape[1], kernel_size=ks, bias=False, padding=padding, stride=stride)
    conv.weight.data = w.data
    input_shape = (1, w.shape[1], input_size, input_size, input_size)

    v = torch.randn(input_shape, device=w.device)
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)

    stop_criterion = False
    it = 0
    while not stop_criterion:
        previous = v
        v = _norm_gradient_sq(conv, v)
        v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_shape)
        stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
        it += 1
    u = conv(Variable(v))  # unormalized left singular vector
    singular_value = torch.norm(u).item()
    return singular_value
