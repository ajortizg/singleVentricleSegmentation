import torch
import time
import torch.nn.functional as F



class Restriction2D_Test1:        
    
    def scale_down(self,u):
        NY = u.size(dim=0)
        NX = u.size(dim=1)

        return F.avg_pool2d(u, kernel_size=3, stride=2, padding=0)

    # def scale_up(self,u):
    #     NY = u.size(dim=0)
    #     NX = u.size(dim=1)
    #     NYUp = 2*NY 
    #     NXUp = 2*NX
        
    #     filter_size = to_size // imageT.size(2)
        
    #     temp_w_inv = torch.zeros([3, 3, NYUp, NXUp])
    #     temp_w_inv[0, 0, :, :] = 1
    #     temp_w_inv[1, 1, :, :] = 1
    #     temp_w_inv[2, 2, :, :] = 1
        
    #     temp_w_inv = Variable(temp_w_inv).type(dtype)
        
    #     return F.conv_transpose2d(u, temp_w_inv, stride=filter_size)