import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.parametrize as P
import utilities.cnn_utils as cnn_utils
import lipschitz as L


class NetC(nn.Module):
    def __init__(self, in_ch, out_ch, ks, padding, stride, in_size, max_lc=1.):
        super(NetC, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=ks, padding=padding, stride=stride)
        P.register_parametrization(self.conv, 'weight', L.L2LipschitzConv3d(in_size, ks, padding, stride, max_lc=max_lc))

    def forward(self, x):
        return F.relu(self.conv(x))


class NetTC(nn.Module):
    def __init__(self, in_ch, out_ch, ks, padding, stride, in_size, max_lc):
        super(NetTC, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, ks, stride, padding)
        P.register_parametrization(self.upsample, 'weight', L.L2LipschitzConvTranspose3d(in_size, ks, padding, stride, max_lc=max_lc))

    def forward(self, x):
        return F.relu(self.upsample(x))


if __name__ == '__main__':
    cnn_utils.seeding(42)
    # input = torch.rand((5, 1, 16, 16, 16))
    # target = torch.rand((5, 1, 8, 8, 8))

    # max_lc = 0.1
    # ks = 3
    # in_size = 16
    # padding = 1
    # stride = 2
    # net = NetC(in_ch=1, out_ch=1, ks=ks, padding=padding, stride=stride, in_size=in_size, max_lc=max_lc)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # loss_fn = torch.nn.MSELoss()

    # for i in range(10):
    #     result = net(input)
    #     loss = loss_fn(result, target)

    #     lc = (torch.tensor(L.spectral_norm_conv3d_2(net.conv.weight, in_size, ks, padding, stride, 1e-8, 1)))
    #     print('epoch: ', i, '  loss: ', loss.item(), '  lipschitz_constant: ', lc.item())
    #     loss.backward()
    #     optimizer.step()

    input = torch.rand((5, 8, 16, 16, 16))
    target = torch.rand((5, 4, 32, 32, 32))

    max_lc = 0.1
    ks = 2
    in_size = 16
    padding = 0
    stride = 2
    net = NetTC(in_ch=8, out_ch=8 // 2, ks=ks, padding=padding, stride=stride, in_size=in_size, max_lc=max_lc)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for i in range(100):
        result = net(input)
        loss = loss_fn(result, target)

        lc = L.spectral_norm_conv_transpose3d_2(net.upsample.weight, in_size, ks, padding, stride, 1e-8, 3)
        print('epoch: ', i, '  loss: ', loss.item(), '  lipschitz_constant: ', lc.item())
        loss.backward()
        optimizer.step()
