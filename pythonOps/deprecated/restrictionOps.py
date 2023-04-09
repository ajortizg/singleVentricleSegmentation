import torch
# import time
# import torch.nn.functional as F


def restrict1d( u ):

  NX = u.shape[0]
  size_reshape = torch.Size([1,NY,NX])
  u_reshaped = torch.reshape(u,size_reshape)

  # u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=0.5, mode='linear', align_corners=False)

  NX_restr = u_restr_reshape.shape[2]
  size_restr = torch.Size([NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr


def restrict2d( u ):

  NY = u.shape[0]
  NX = u.shape[1]
  size_reshape = torch.Size([1,1,NY,NX])
  u_reshaped = torch.reshape(u,size_reshape)

#   u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=0.5, mode='bilinear', align_corners=False)

  NY_restr = u_restr_reshape.shape[2]
  NX_restr = u_restr_reshape.shape[3]
  size_restr = torch.Size([NY_restr,NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr


def restrict3d( u ):
  
#   sizeRestr = u_restr.size()

  NZ = u.shape[0]
  NY = u.shape[1]
  NX = u.shape[2]
  size_reshape = torch.Size([1,1,NZ,NY,NX])
  u_reshaped = torch.reshape(u,size_reshape)

#   u_restr = torch.nn.functional.interpolate(u, size=sizeRestr, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  u_restr_reshape = torch.nn.functional.interpolate(u_reshaped, scale_factor=0.5, mode='trilinear', align_corners=False)

  NZ_restr = u_restr_reshape.shape[2]
  NY_restr = u_restr_reshape.shape[3]
  NX_restr = u_restr_reshape.shape[4]
  size_restr = torch.Size([NZ_restr,NY_restr,NX_restr])

  u_restr = torch.reshape(u_restr_reshape, size_restr)

#   print("u reshaped size = ", size_restr)
#   print("max norm u_restr_reshape = ", torch.max(torch.abs(u_restr_reshape)).item() )
#   print("max norm u_restr = ", torch.max(torch.abs(u_restr)).item() )

  return u_restr



# class Restriction2D_Test1:        
    
#     def scale_down(self,u):
#         NY = u.size(dim=0)
#         NX = u.size(dim=1)

#         return F.avg_pool2d(u, kernel_size=3, stride=2, padding=0)

#     # def scale_up(self,u):
#     #     NY = u.size(dim=0)
#     #     NX = u.size(dim=1)
#     #     NYUp = 2*NY 
#     #     NXUp = 2*NX
        
#     #     filter_size = to_size // imageT.size(2)
        
#     #     temp_w_inv = torch.zeros([3, 3, NYUp, NXUp])
#     #     temp_w_inv[0, 0, :, :] = 1
#     #     temp_w_inv[1, 1, :, :] = 1
#     #     temp_w_inv[2, 2, :, :] = 1
        
#     #     temp_w_inv = Variable(temp_w_inv).type(dtype)
        
#     #     return F.conv_transpose2d(u, temp_w_inv, stride=filter_size)