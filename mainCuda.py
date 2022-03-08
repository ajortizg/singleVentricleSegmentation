import nibabel as nib
import numpy as np
import os

from utils.config import *
# from tvl1.tvl1Cuda import TVL1Cuda
from tvl1.tvl1Cuda import *

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

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

    #convert to torch for given time steps
    t0, t1 = 8, 9
    I0 = torch.from_numpy(nii_data[:,:,:,t0]).float().to(DEVICE)
    I1 = torch.from_numpy(nii_data[:,:,:,t1]).float().to(DEVICE)

    #initialization of optical flow 
    u = torch.zeros([NZ,NY,NX,3]).float().to(DEVICE)
    p = torch.zeros([NZ,NY,NX,3,3]).float().to(DEVICE)

    # Compute the optical flow
    alg = TVL1OpticalFlowCuda()
    alg.computeOnPyramid(I0, I1, u, p)


    # #TODO swap result (Z,Y,X) back to (X,Y,Z):
    # #result_backSwap = np.swapaxes(result, 0, 2)
