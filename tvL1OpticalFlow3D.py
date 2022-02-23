#==================================
import sys
sys.path.append('core')

#==================================
import argparse
import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib #to save from matplotlib to tikz
import math
import torch
import time
import torch.nn.functional as F

#==================================
import differentialOps

from tvl1OF3d_cuda_ext import tvl1OF3d


#==================================
DEVICE = 'cuda'


# def prox_ind(p, lamda):
#     p_norm = torch.sqrt((p**2).sum(3,keepdims=True))
#     return p/torch.maximum(p_norm/lamda,torch.ones_like(p_norm))

# def primal_step(u, p, ATq, tau, hz):
#     u_tmp = u - tau*(backward_torch(p, hz) + ATq)
#     return u_tmp

# def dual_step(p, u, hz, sigma, lamda):
#     p_tmp = p + sigma[None,None,None,:]*forward_torch(u, hz)
#     return prox_ind(p_tmp, lamda)

# Z = 100
# M = 200
# N = 200

# ####################################################################
# # test primal update step
# print('\ntest primal update step: ')

# p = torch.randn(Z, M, N, 3).cuda()
# u = torch.randn(Z, M, N).cuda()
# ATq = torch.randn(Z, M, N).cuda()
# tau = 0.001 #tau = torch.randn(Z, M, N).cuda() #
# hz = 0.5

# ts = time.time()
# # pytorch result
# u_old = u.clone()
# u_update_torch = primal_step(u, p, ATq, tau, hz)
# print('primal - u unchanged by pytorch op: ', torch.allclose(u_old, u))
# print('primal - elapsed time pytorch: ', (time.time()-ts))

# ts = time.time()
# # cuda kernel result
# reco3d_tv.primal_step(u, p, ATq, tau, hz)
# print('primal - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('primal - Check diff with infty norm: ', torch.max(torch.abs(u_update_torch - u)).item())

# ####################################################################
# # test dual update step
# print('\ntest dual update step: ')

# sigma = torch.Tensor([0.5, 0.5, 0.25]).cuda()
# lamda = 0.01
# hz = 0.5
# ts = time.time()
# # pytorch result
# p_old = p.clone()
# p_update_torch = dual_step(p, u, hz, sigma, lamda)
# print('dual - p unchanged by pytorch op: ', torch.allclose(p_old, p))
# print('dual - elapsed time pytorch: ', (time.time()-ts))
# ts = time.time()
# # cuda kernel result
# reco3d_tv.dual_step(p, u, sigma, hz, lamda)
# print('dual - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('dual - Check diff with infty norm: ', torch.max(torch.abs(p_update_torch - p)).item())

# ####################################################################
# # test prox l2
# print('\ntest prox l2: ')

# q = torch.randn(Z, M, N).cuda()
# sigma = torch.abs(torch.randn(Z)).cuda()

# ts = time.time()
# # pytorch result
# sigma_old = sigma.clone()
# q_torch = q/(1+sigma[:,None,None])
# print('prox l2 - sigma remains unchanged by pytorch op: ', torch.allclose(sigma_old, sigma))
# print('prox l2 - elapsed time pytorch: ', (time.time()-ts))

# ts = time.time()
# # cuda kernel result
# reco3d_tv.prox_l2(q, sigma)
# print('prox l2 - elapsed time cuda kernel: ', (time.time()-ts))

# # infinity norm to measure difference
# print('prox l2 - Check diff with infty norm: ', torch.max(torch.abs(q_torch - q)).item())



def computeTVL1OpticalFlow3D(args):

    print("""
    ==================================
        computeTVL1OpticalFlow3D
    ==================================
    """)

    #==================================
    # Load 4D nifty [x,y,z,t]
    print("\n=================\nLoad Nifty File\n=================")
    if not (args.fileName):
       parser.error('add -fileName')
    fileName = args.fileName

    nii_img = nib.load(fileName)
    nii_data_xyzt = nii_img.get_fdata()

    NX = nii_data_xyzt.shape[0]
    NY = nii_data_xyzt.shape[1]
    NZ = nii_data_xyzt.shape[2]
    NT = nii_data_xyzt.shape[3]
    print( f"dimension of input: (X,Y,Z,T) = {NX,NY,NZ,NT}")


    ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    print( f"dimension after swap: (Z,Y,X,T) = {nii_data.shape}")

    #==================================
    #scaling of data 
    totalMinValue = np.amin(nii_data)
    totalMaxValue = np.amax(nii_data)
    print(f"input (min,max) = {totalMinValue,totalMaxValue}")
    scaleMaxValue = 255.
    print("scaling of data to max value", scaleMaxValue)
    nii_data *= scaleMaxValue / totalMaxValue
    

    #==================================
    # save directory
    saveDir = os.path.dirname("results") 
    if args.saveDir is not None:
        saveDir = os.path.dirname(args.saveDir) 
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    
    #==================================
    # compute for first slice
    tA = 8
    tB = 9
    imageA = torch.from_numpy(nii_data[:,:,:,tA]).float().to(DEVICE)
    imageB = torch.from_numpy(nii_data[:,:,:,tB]).float().to(DEVICE)


    #TODO swap result (Z,Y,X) back to (X,Y,Z):
    #result_backSwap = np.swapaxes(result, 0, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', help="file name of 4d nifty")
    parser.add_argument('--saveDir', help="directory for saving")

    args = parser.parse_args()


    computeTVL1OpticalFlow3D(args)