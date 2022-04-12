
import sys
import numpy as np
import nibabel as nib
import os
import pandas
import configparser
import torch

utils_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_lib_path)
import plots

from opticalFlow_cuda_ext import opticalFlow


def save_torch_to_nifty(file,saveDir,fileName,hdr_old,zooms="old"):
    #convert
    file_np = file.cpu().detach().numpy()
    file_xyzt = np.swapaxes(file_np, 0, 2)
    #header 
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform( hdr_old.get_qform() )
    hdr.set_sform( hdr_old.get_sform() )
    if zooms == "old":
      hdr.set_zooms( hdr_old.get_zooms() )
    else: 
      hdr.set_zooms( zooms  )
    #img
    ni_img = nib.Nifti1Image(file_xyzt, affine=None, header=hdr)
    #save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)
    # print("old header:")
    # print(hdr_old) 
    # print("new header:")
    # print(hdr) 


def getMeshLength(config,NZ,NY,NX):
    LenghtType = config.get('PARAMETERS', 'LenghtType')
    if LenghtType == "numDofs":
        LZ = NZ-1
        LY = NY-1
        LX = NX-1
        return LZ, LY, LX
    elif LenghtType == "fixed":
        LZ = config.getfloat('PARAMETERS', "LenghtZ")
        LY = config.getfloat('PARAMETERS', "LenghtY")
        LX = config.getfloat('PARAMETERS', "LenghtX")
        return LZ, LY, LX

if __name__ == "__main__":

    print("\n\n")
    print("==================================================")
    print("==================================================")
    print("  preprocessing data: prolongation                ")
    print("==================================================")
    print("==================================================")
    print("\n\n")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configTVL1OF3D.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe else "cpu"

    # interpolation
    interType = config.get('PARAMETERS', 'InterpolationType')
    InterpolationTypeCuda = None
    if interType == "NEAREST":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
    elif interType == "LINEAR":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
    elif interType == "CUBIC_HERMITESPLINE":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
    else:
        raise Exception("wrong InterpolationType in configParser")
    #boundary
    boundaryType = config.get('PARAMETERS', 'BoundaryType')
    BoundaryTypeCuda = None
    if boundaryType == "NEAREST":
        BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_NEAREST
    elif boundaryType == "MIRROR":
        BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_MIRROR
    elif boundaryType == "REFLECT":
        BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_REFLECT
    else:
        raise Exception("wrong BoundaryType in configParser")

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing3D_prolongation" )

    #save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
      config.write(configfile)

    # load data base
    BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
    VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
    VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
    SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
    SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
    df = pandas.read_excel(SEGMENTATIONS_FILE)
    numDataFiles = df.shape[0]
    print("number of data files = ", numDataFiles)
    SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
    SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])

    #
    saveDir4D = plots.createSubDirectory(saveDir, VOLUMES_SUBDIR_PATH)
    saveDirSegmentations = plots.createSubDirectory(saveDir, SEGMENTATIONS_SUBDIR_PATH)

    for index, row in df.iterrows():

        # Load 4D nifty [x,y,z,t]
        print("=======================================")
        PATIENT_NAME = row['Name']
        print("load data for patient: ", PATIENT_NAME)
        vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
        vol_hdr = vol.header
        vol_affine = vol.affine
        nii_data_xyzt = vol.get_fdata()
        NX = nii_data_xyzt.shape[0]
        NY = nii_data_xyzt.shape[1]
        NZ = nii_data_xyzt.shape[2]
        NT = nii_data_xyzt.shape[3]
        print("   * (NX,NY,NZ,NT) = ", NX, NY, NZ, NT )
        nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
        data_4d = torch.from_numpy(nii_data).float().to(DEVICE)
        #
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, PATIENT_NAME)
        #read time steps for diastole and systole
        tDiastole = row["Diastole"]
        tSystole = row["Systole"]
        print("   * systole at time:  ", tSystole)
        print("   * diastole at time: ", tDiastole)
        print("=======================================")


        #read zooms from nifty file
        zooms = vol_hdr.get_zooms()
        zoomX = round(zooms[0])
        zoomY = round(zooms[1])
        zoomZ = round(zooms[2])
        zoomT = zooms[3]

        #generate old mesh and new mesh for prolongation 
        LZ,LY,LX = getMeshLength(config,NZ,NY,NX)
        meshInfo_old = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

        NZ_prolong,NY_prolong,NX_prolong = zoomZ*NZ,zoomY*NY,zoomX*NX
        LZ_prolong,LY_prolong,LX_prolong = getMeshLength(config,NZ_prolong,NY_prolong,NX_prolong)
        meshInfo_new = opticalFlow.MeshInfo3D(NZ_prolong,NY_prolong,NX_prolong,LZ_prolong,LY_prolong,LX_prolong)
        prolongationOp = opticalFlow.Prolongation3D(meshInfo_old,meshInfo_new,InterpolationTypeCuda,BoundaryTypeCuda)

        # get input masks for diastole
        nii_mask_diastole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
        hdr_mask_diastole = nii_mask_diastole_load.header
        affine_mask_diastole = nii_mask_diastole_load.affine
        nii_mask_diastole_xyz = nii_mask_diastole_load.get_fdata()
        nii_mask_diastole = np.swapaxes(nii_mask_diastole_xyz, 0, 2)
        mask_diastole = torch.from_numpy(nii_mask_diastole).float().to(DEVICE)

        # get input masks for systole 
        nii_mask_systole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
        hdr_mask_systole = nii_mask_systole_load.header
        affine_mask_systole = nii_mask_systole_load.affine
        nii_mask_systole_xyz = nii_mask_systole_load.get_fdata()
        nii_mask_systole = np.swapaxes(nii_mask_systole_xyz, 0, 2)
        mask_systole = torch.from_numpy(nii_mask_systole).float().to(DEVICE)

        #prolongate 
        prolongation_diastole = prolongationOp.forward(mask_diastole)
        prolongation_systole = prolongationOp.forward(mask_systole)
        prolongation_4d = prolongationOp.forwardVectorField(data_4d.contiguous())

        # save as nifty
        save_torch_to_nifty(prolongation_4d, saveDir4D, PATIENT_NAME + ".nii.gz", vol_hdr, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong, zoomT) )
        save_torch_to_nifty(prolongation_diastole, saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii", hdr_mask_diastole, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
        save_torch_to_nifty(prolongation_systole, saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii", hdr_mask_systole, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )


        # #convert
        # prolongation_4d_np = prolongation_4d.cpu().detach().numpy()
        # prolongation_4d_xyzt = np.swapaxes(prolongation_4d_np, 0, 2)
        # #header 
        # ni_img_4d_hdr = nib.nifti1.Nifti1Header()
        # ni_img_4d_hdr.set_data_shape((NX_prolong,NY_prolong,NZ_prolong,NT))
        # ni_img_4d_hdr.set_qform( vol_hdr.get_qform() )
        # ni_img_4d_hdr.set_sform( vol_hdr.get_sform() )
        # ni_img_4d_hdr.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong, zoomT)  )
        # #img
        # ni_img_4d = nib.Nifti1Image(prolongation_4d_xyzt, affine=None, header=ni_img_4d_hdr)
        # #save
        # outputFile_4d = os.path.sep.join([saveDir4D, PATIENT_NAME + ".nii.gz"])
        # nib.save(ni_img_4d, outputFile_4d)

        # #save diastole as nifty
        # #convert 
        # prolongation_diastole_np = prolongation_diastole.cpu().detach().numpy()
        # prolongation_diastole_xyzt = np.swapaxes(prolongation_diastole_np, 0, 2)
        # #header 
        # hdr_mask_diastole_prolong = nib.nifti1.Nifti1Header()
        # hdr_mask_diastole_prolong.set_data_shape((NX_prolong,NY_prolong,NZ_prolong))
        # hdr_mask_diastole_prolong.set_qform( hdr_mask_diastole.get_qform() )
        # hdr_mask_diastole_prolong.set_sform( hdr_mask_diastole.get_sform() )
        # hdr_mask_diastole_prolong.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
        # #img
        # ni_img_diastole = nib.Nifti1Image(prolongation_diastole_xyzt, affine=None, header=hdr_mask_diastole_prolong)
        # #save
        # outputFile_diastole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii"])
        # nib.save(ni_img_diastole, outputFile_diastole)

        # #save systole as nifty
        # #convert 
        # prolongation_systole_np = prolongation_systole.cpu().detach().numpy()
        # prolongation_systole_xyzt = np.swapaxes(prolongation_systole_np, 0, 2)
        # #header 
        # hdr_mask_systole_prolong = nib.nifti1.Nifti1Header()
        # hdr_mask_systole_prolong.set_data_shape((NX_prolong,NY_prolong,NZ_prolong))
        # hdr_mask_systole_prolong.set_qform( hdr_mask_systole.get_qform() )
        # hdr_mask_systole_prolong.set_sform( hdr_mask_systole.get_sform() )
        # hdr_mask_systole_prolong.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
        # #img
        # ni_img_systole = nib.Nifti1Image(prolongation_systole_xyzt, affine=None, header=hdr_mask_systole_prolong)
        # #save
        # outputFile_systole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii"])
        # nib.save(ni_img_systole, outputFile_systole)

    #save data base
    output_df = os.path.sep.join([saveDir, SEGMENTATIONS_FILE_NAME])
    df.to_excel(output_df)  