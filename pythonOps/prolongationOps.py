import torch
# import time
# import torch.nn.functional as F


def prolongate1d( u ):

  NX = u.shape[0]
  size_reshape = torch.Size([1,NX])
  u_reshaped = torch.reshape(u,size_reshape)

  # u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=2., mode='linear', align_corners=False)

  NX_restr = u_restr_reshape.shape[2]
  size_restr = torch.Size([NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr


def prolongate2d( u ):

  NY = u.shape[0]
  NX = u.shape[1]
  size_reshape = torch.Size([1,1,NY,NX])
  u_reshaped = torch.reshape(u,size_reshape)

#   u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=2., mode='bilinear', align_corners=False)

  NY_restr = u_restr_reshape.shape[2]
  NX_restr = u_restr_reshape.shape[3]
  size_restr = torch.Size([NY_restr,NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr


def prolongate3d( u ):

  NZ = u.shape[0]
  NY = u.shape[1]
  NX = u.shape[2]
  size_reshape = torch.Size([1,1,NZ,NY,NX])
  u_reshaped = torch.reshape(u,size_reshape)

#   u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=2., mode='trilinear', align_corners=False)

  NZ_restr = u_restr_reshape.shape[2]
  NY_restr = u_restr_reshape.shape[3]
  NX_restr = u_restr_reshape.shape[4]
  size_restr = torch.Size([NZ_restr,NY_restr,NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr
