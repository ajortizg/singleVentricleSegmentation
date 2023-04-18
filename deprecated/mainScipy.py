import nibabel as nib
import numpy as np
import time
import os

from utilities.config import *
from TVL1OF.tvl1Scipy import TVL1Scipy

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    # Load 4D nifty [x,y,z,t]
    vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_FILE]))
    data = vol.get_fdata()
    print(f"Dims: {data.ndim}, shape: {data.shape}, type: {data.dtype}")

    #==================================
    # Load 4D nifty [x,y,z,t]
    vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_FILE]))
    nii_data_xyzt = vol.get_fdata()
    print(f"Dims: {nii_data_xyzt.ndim}, shape: {nii_data_xyzt.shape}, type: {nii_data_xyzt.dtype}")
    NX = nii_data_xyzt.shape[0]
    NY = nii_data_xyzt.shape[1]
    NZ = nii_data_xyzt.shape[2]
    NT = nii_data_xyzt.shape[3]

    #==================================
    #scaling of data 
    totalMinValue = np.amin(nii_data_xyzt)
    totalMaxValue = np.amax(nii_data_xyzt)
    print(f"input (min,max) = {totalMinValue,totalMaxValue}")
    # scaleMaxValue = 255.
    scaleMaxValue = 1.
    print("scaling of data to max value", scaleMaxValue)
    nii_data_xyzt *= scaleMaxValue / totalMaxValue

    ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    print( f"dimension after swap: (Z,Y,X,T) = {nii_data.shape}")

    # Get two adjacent volumes and change axes to z,y,x
    t0, t1 = 8, 9
    # I0 = nii_data[:12, :, :, t0]
    # I1 = nii_data[:12, :, :, t1]
    I0 = nii_data[:, :, :, t0]
    I1 = nii_data[:, :, :, t1]

    #
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, timestr])
    if not os.path.exists(saveDir):
      os.makedirs(saveDir)

    # Compute the optical flow
    alg = TVL1Scipy(saveDir)
    alg.compute(I0, I1)
