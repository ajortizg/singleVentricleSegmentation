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
sys.path.append("../pythonOps/")
import restrictionOps

sys.path.append("../utils/")
import plots

from opticalFlow_cuda_ext import opticalFlow


#==================================
DEVICE = 'cuda'



def checkRestrictionOps(args):

    print("""
    ==================================
        checkRestrictionOps
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
    # check in 2D
    tA = 8
    tB = 9
    z=8
    imageA = torch.from_numpy(nii_data[z,:,:,tA]).float().to(DEVICE)
    imageB = torch.from_numpy(nii_data[z,:,:,tB]).float().to(DEVICE)

    

    
    # sizeImageA_down = torch.Size([NY//2,NX//2])
    # imageA_down = torch.zeros((sizeImageA_down)).cuda()
    # restrictionOps.restrict2d(imageA,imageA_down)
    imageA_down = restrictionOps.restrict2d(imageA)

    # save 
    print("imageA.shape = ", imageA.shape )
    print("imageA_down.shape = ", imageA_down.shape )

    imageA_np = imageA.cpu().detach().numpy()
    plots.saveImage(imageA_np,saveDir,"imgA2D.png")

    imageA_down_np = imageA_down.cpu().detach().numpy()
    plots.saveImage(imageA_down_np,saveDir,"imgA2D_down.png")

    #TODO swap result (Z,Y,X) back to (X,Y,Z):
    #result_backSwap = np.swapaxes(result, 0, 2)



    #==================================
    # check in 3D
    # tA = 8
    # volA = torch.from_numpy(nii_data[:,:,:,tA]).float().to(DEVICE)
    # volA_down = restrictionOps.restrict3d(volA)

    # # save 
    # print("imageA.shape = ", imageA.shape )
    # print("imageA_down.shape = ", imageA_down.shape )

    # volA_np = volA.cpu().detach().numpy()
    # plots.plot_3d(volA_np) #,saveDir,"volA3D.png")

    # volA_down_np = volA_down.cpu().detach().numpy()
    # plots.plot_3d(volA_down_np) #,saveDir,"volA2D_down.png")

    #TODO swap result (Z,Y,X) back to (X,Y,Z):
    #result_backSwap = np.swapaxes(result, 0, 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', help="file name of 4d nifty")
    parser.add_argument('--saveDir', help="directory for saving")

    args = parser.parse_args()


    checkRestrictionOps(args)