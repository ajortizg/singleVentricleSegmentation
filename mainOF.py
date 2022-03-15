import nibabel as nib
import numpy as np
import os
import pandas

from utils.config import *
from tvl1.TVL1OF import *

from opticalFlow_cuda_ext import opticalFlow


if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("        compute TV-L1 optical flow:")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # create save directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, timestr])
    if not os.path.exists(saveDir):
      os.makedirs(saveDir)
    print("save results to directory: ", saveDir, "\n")
    #np.set_printoptions(precision=2, suppress=True)

    # Load 4D nifty [x,y,z,t]
    print("=======================================")
    print("load data for patient: ", PATIENT_NAME)
    vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
    nii_data_xyzt = vol.get_fdata()
    NX = nii_data_xyzt.shape[0]
    NY = nii_data_xyzt.shape[1]
    NZ = nii_data_xyzt.shape[2]
    NT = nii_data_xyzt.shape[3]

    #==================================
    #scaling of data 
    totalMinValue = np.amin(nii_data_xyzt)
    totalMaxValue = np.amax(nii_data_xyzt)
    scaleMinValue = 0.
    scaleMaxValue = 1.
    # scaleMaxValue = 255.
    print(f"   * scaling of data in range {totalMinValue,totalMaxValue} to {scaleMinValue,scaleMaxValue}")
    nii_data_xyzt *= scaleMaxValue / totalMaxValue

    ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    #print("swap axes (X,Y,Z,T) to (Z,Y,X,T)")
    nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    print( f"   * dimensions: (Z,Y,X,T) = {nii_data.shape}")

    #read time steps for diastole and systole
    df = pandas.read_excel(SEGMENTATIONS_FILE)
    rowPatient = df[df['Name'] == PATIENT_NAME]
    indexPatient = rowPatient.index[0]
    tDiastole = rowPatient.loc[indexPatient, "Diastole"]
    tSystole = rowPatient.loc[indexPatient, "Systole"]
    print("   * systole at time:  ", tSystole)
    print("   * diastole at time: ", tDiastole)
    numTimeSteps = tDiastole - tSystole
    print("=======================================")
    print("\n")

    #for t in range(tSystole, tDiastole):
    for t in range(tSystole, tSystole+3):
      saveDirTimeStep = os.path.sep.join([saveDir, f"time{t}"])
      if not os.path.exists(saveDirTimeStep):
        os.makedirs(saveDirTimeStep)
      #convert to torch for given time steps
      t0, t1 = t, t+1
      I0 = torch.from_numpy(nii_data[:,:,:,t0]).float().to(DEVICE)
      I1 = torch.from_numpy(nii_data[:,:,:,t1]).float().to(DEVICE)

      #initialization of optical flow 
      u = torch.zeros([NZ,NY,NX,3]).float().to(DEVICE)
      p = torch.zeros([NZ,NY,NX,3,3]).float().to(DEVICE)

      # Compute the optical flow
      alg = TVL1OpticalFlow(saveDirTimeStep)
      alg.computeOnPyramid(I0, I1, u, p)
      #alg.computeOnPyramid(I1, I0, u, p)


    # warp the input mask with the computed optical flow 
    saveDirTimeSystole = os.path.sep.join([saveDir, f"time{tSystole}"])
    saveDirStep = os.path.sep.join([saveDirTimeSystole, f"it0"])
    nii_mask_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
    nii_mask_xyz = nii_mask_load.get_fdata()
    ## swap from nibabel (X,Y,Z) to cuda-compatible (Z,Y,X):
    print("swap axes (X,Y,Z) to (Z,Y,X)")
    nii_mask = np.swapaxes(nii_mask_xyz, 0, 2)
    # nii_mask = np.swapaxes(nii_mask_zyx, 1, 2)
    print( f"dimension after swap: (Z,Y,X) = {nii_mask.shape}")
    mask = torch.from_numpy(nii_mask).float().to(DEVICE)
    save_slices(mask, f"mask_Systole.png", saveDirStep)
    save_single_zslices(mask, saveDirStep, "mask_slices", 1., 2)

    flowName = "flow_it0.pt"
    fileNameFlow = os.path.join(saveDirStep, flowName) 
    u = torch.load(fileNameFlow, map_location=torch.device(DEVICE))

    NZ, NY, NX = mask.shape[0], mask.shape[1], mask.shape[2]
    LZ, LY, LX = NZ-1, NY-1, NX-1
    meshInfo3D_cuda = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)
    warpingOp = opticalFlow.Warping3D(meshInfo3D_cuda)
    mask_warped = warpingOp.forward(mask,u,opticalFlow.InterpolationType.INTERPOLATE_LINEAR)
    #mask_warped = warpingOp.forward(mask,u,opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE)

    save_slices(mask_warped, f"mask_warped_time{t0}.png", saveDirStep)
    save_single_zslices(mask_warped, saveDirStep, "mask_warped_slices", 1., 2)

    #add mask to mri images 
    saveDirMRISlices = os.path.join(saveDirStep, "I0Slices")
    saveDirMaskSlices = os.path.join(saveDirStep, "mask_slices")
    saveDirSumSlices = os.path.join(saveDirStep, "sum_slices")
    if not os.path.exists(saveDirSumSlices):
        os.makedirs(saveDirSumSlices)
    for z in range(0,NZ):
      fileNameMRISlice = os.path.join(saveDirMRISlices, f"colorimg_z{z}.png") 
      img_mri = cv2.imread(fileNameMRISlice)
      fileNameMaskSlice = os.path.join(saveDirMaskSlices, f"colorimg_z{z}.png") 
      img_mask = cv2.imread(fileNameMaskSlice)
      fileNameSumSlice = os.path.join(saveDirSumSlices, f"sumimg_z{z}.png")
      img_sum = img_mri + img_mask 
      cv2.imwrite(fileNameSumSlice,img_sum)
      fileNameSumSliceInvert = os.path.join(saveDirSumSlices, f"invertsumimg_z{z}.png")
      img_sum_invert = 255. - img_sum
      cv2.imwrite(fileNameSumSliceInvert,img_sum_invert)


    # #TODO swap result (Z,Y,X) back to (X,Y,Z):
    # #result_backSwap = np.swapaxes(result, 0, 2)
