import nibabel as nib
import numpy as np
import os
import pandas
from PIL import Image

from utils.config import *
from tvl1.TVL1OF_2D import *

from opticalFlow_cuda_ext import opticalFlow


if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("        compute TV-L1 optical flow in 2D:")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # create save directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, "TVL1OF2D_" + timestr])
    if not os.path.exists(saveDir):
      os.makedirs(saveDir)
    print("save results to directory: ", saveDir, "\n")
    #np.set_printoptions(precision=2, suppress=True)

    # Load 2D images [x,y,z,t]
    print("=======================================")
    # print("load data for patient: ", PATIENT_NAME)
    vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
    nii_data_xyzt = vol.get_fdata()

    image0 = Image.open(Image0_PATH)
    image1 = Image.open(Image1_PATH)
    
    # summarize some details about the image
    #print(image0.format)
    #print(image0.size)
    #print(image0.mode)

    NX = nii_data_xyzt.shape[0]
    NY = nii_data_xyzt.shape[1]

    # #==================================
    # #scaling of data 
    totalMinValue = min(np.amin(image0),np.amin(image1))
    totalMaxValue = max(np.amax(image0),np.amax(image0))
    scaleMinValue = 0.
    scaleMaxValue = 1.
    # scaleMaxValue = 255.
    print(f"   * scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
    image0 *= scaleMaxValue / totalMaxValue
    image1 *= scaleMaxValue / totalMaxValue

    ## swap from (X,Y) to cuda-compatible (Y,X):
    I0np = np.swapaxes(image0, 0, 1)
    I1np = np.swapaxes(image1, 0, 1)
    print( f"   * dimensions I0: (Y,X) = {I0np.shape}")
    print( f"   * dimensions I1: (Y,X) = {I1np.shape}")

    #convert to torch for given time steps
    I0 = torch.from_numpy(I0np).float().to(DEVICE)
    I1 = torch.from_numpy(I1np).float().to(DEVICE)
    # I0 = torch.from_numpy(image0).float().to(DEVICE)
    # I1 = torch.from_numpy(image1).float().to(DEVICE)
    # I0 = torch.swapaxes(I0, 0, 1)
    # I1 = torch.swapaxes(I1, 0, 1)

    #initialization of optical flow 
    u = torch.zeros([NY,NX,2]).float().to(DEVICE)
    p = torch.zeros([NY,NX,2,2]).float().to(DEVICE)

    # Compute the optical flow
    alg = TVL1OpticalFlow2D(saveDir)
    alg.computeOnPyramid(I0, I1, u, p)
    #alg.computeOnPyramid(I1, I0, u, p)


