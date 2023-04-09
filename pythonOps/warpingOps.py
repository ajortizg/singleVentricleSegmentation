import numpy as np
import math
import torch
# from colorama import init
from termcolor import colored
from scipy import ndimage



def interpolate1d_linear( u, meshInfo1D, coord_x_warped):
  ix_f = math.floor(coord_x_warped / meshInfo1D.hX)
  ix_c = ix_f + 1
  wx = coord_x_warped / meshInfo1D.hX - ix_f

  u_f = 0
  if ix_f >= 0 and ix_f < meshInfo1D.NX:
    u_f = u[ix_f]

  u_c = 0;
  if ix_c >= 0 and ix_c < meshInfo1D.NX:
    u_c = u[ix_c]

  out = (1 - wx) * u_f
  out += wx * u_c

  return out


def warp1d_linear( u, phi, meshInfo1D, u_warped):
  for ix in range(0,meshInfo1D.NX):
    coord_x_warped = ix * meshInfo1D.hX + phi[ix,0]
    u_warped[ix] = interpolate1d_linear(u, meshInfo1D, coord_x_warped)


def interpolate1d_cubic_local(localBuffer, local_coord):

  kernel_size=4

  ix_f = math.floor(local_coord)
  ix_c = ix_f + 1
  ix_f_1 = ix_f - 1
  ix_c_1 = ix_c + 1

  i_f = i_f_1 = i_c = i_c_1  = 0
  if ix_f >= 0 and ix_f < kernel_size:
    i_f = localBuffer[ix_f]
  if ix_f_1 >= 0 and ix_f_1 < kernel_size:
    i_f_1 = localBuffer[ix_f_1]
  if ix_c >= 0 and ix_c < kernel_size: 
    i_c = localBuffer[ix_c]
  if ix_c_1 >= 0 and ix_c_1 < kernel_size:
    i_c_1 = localBuffer[ix_c_1]

  # determine the coefficients
  p_f = i_f
  p_prime_f = (i_c - i_f_1) / 2
  p_c = i_c
  p_prime_c = (i_c_1 - i_f) / 2

  a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c
  b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c
  c = p_prime_f
  d = p_f

  wx = local_coord - ix_f

  out = wx * (wx * (wx * a + b) + c) + d

  return out


def interpolate1d_cubic(u, meshInfo1D, coord_x_warped):

  ix_f = math.floor(coord_x_warped / meshInfo1D.hX)
  wx = coord_x_warped / meshInfo1D.hX - ix_f
  buff_x = np.empty(4)

  for dx in range(-1,3):
    c_ix_x = ix_f + dx
    if c_ix_x >= 0 and c_ix_x < meshInfo1D.NX:
      buff_x[dx + 1] = u[c_ix_x]
    else:
      buff_x[dx + 1] = 0

  out = interpolate1d_cubic_local(buff_x, wx + 1)

  return out


def warp1d_cubic( u, phi, meshInfo1D, u_warped):
  for ix in range(0,meshInfo1D.NX):
    coord_x_warped = ix * meshInfo1D.hX + phi[ix,0]
    u_warped[ix] = interpolate1d_cubic(u, meshInfo1D, coord_x_warped)



class Warping1D:      

    def __init__(self, meshInfo1D):
        self.meshInfo = meshInfo1D
    
    def forward(self, u, phi, interpolationType):
        device = u.device
        NX = u.size(dim=0)
        u_warped = torch.zeros((NX), device=device)
        if interpolationType == "linear":
          warp1d_linear( u, phi, self.meshInfo, u_warped)
        elif interpolationType == "cubic":
          warp1d_cubic( u, phi, self.meshInfo, u_warped)
        else:
          print("Error: wrong interpolation type for warping")

        return u_warped
        
    # def backward(self, p):
    #     return u






# WARPING with pytorch interpolate
#     NY = u.shape[0]
#     NX = u.shape[1]

#     x: [B, C, H, W] (im2)
#     flo: [B, 2, H, W] flow

#     B, C, H, W = x.size()
#     # mesh grid
#     xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
#     yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
#     xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
#     yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
#     grid = torch.cat((xx ,yy) ,1).float()

#     if x.is_cuda:
#         grid = grid.cuda()
#     vgrid = Variable(grid) + flo

#     # scale grid to [-1,1]
#     vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
#     vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

#     vgrid = vgrid.permute(0 , 2 ,3 ,1)
#     flo = flo.permute(0 , 2 , 3 , 1)
#     output = F.grid_sample(x, vgrid)
#     mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
#     mask = F.grid_sample(mask, vgrid)

#     mask[mask <0.9999] = 0
#     mask[mask >0] = 1

#     return output*mask

# torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)