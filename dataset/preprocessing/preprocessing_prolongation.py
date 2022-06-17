
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas
import configparser
import torch
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utils import plots
from dataset import singleVentricleDataset

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

    plots.printConsoleOutput_Header("preprocessing data: prolongation")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')
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

    # prolongation size
    NX_prolong = config.getint('PROLONGATION', 'NX_prolong')
    NY_prolong = config.getint('PROLONGATION', 'NY_prolong')
    NZ_prolong = config.getint('PROLONGATION', 'NZ_prolong')
    LZ_prolong,LY_prolong,LX_prolong = getMeshLength(config,NZ_prolong,NY_prolong,NX_prolong)

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_prolongation" )

    #save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
      config.write(configfile)

    # # load data base
    # BASE_PATH_3D = config.get('DATA', 'BASE_PATH_3D')
    # VOLUMES_SUBDIR_PATH = config.get('DATA', 'VOLUMES_SUBDIR_PATH')
    # VOLUMES_PATH = os.path.sep.join([BASE_PATH_3D, VOLUMES_SUBDIR_PATH])
    # SEGMENTATIONS_FILE_NAME = config.get('DATA', 'SEGMENTATIONS_FILE_NAME')
    # SEGMENTATIONS_FILE = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_FILE_NAME])
    # df = pandas.read_excel(SEGMENTATIONS_FILE)
    # numDataFiles = df.shape[0]
    # print("number of data files = ", numDataFiles)
    # SEGMENTATIONS_SUBDIR_PATH = config.get('DATA', 'SEGMENTATIONS_SUBDIR_PATH')
    # SEGMENTATIONS_PATH = os.path.sep.join([BASE_PATH_3D, SEGMENTATIONS_SUBDIR_PATH])
    # load data base
    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    #
    saveDir4D = plots.createSubDirectory(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dataSet.segmentations_subdir_path)

    #generate columns for prolongation factors
    xprolongfac = np.zeros(len(dataSet))
    yprolongfac = np.zeros(len(dataSet))
    zprolongfac = np.zeros(len(dataSet))


    #iterate over all patients
    pbar = tqdm(total=len(dataSet))
    for index in range(0,len(dataSet)):

        patient = dataSet[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        #read zooms from nifty file
        zooms = patient.nii_header_xyzt.get_zooms()
        # zoomX = round(zooms[0])
        # zoomY = round(zooms[1])
        # zoomZ = round(zooms[2])
        zoomX = zooms[0]
        zoomY = zooms[1]
        zoomZ = zooms[2]
        zoomT = zooms[3]
        # NZ_prolong,NY_prolong,NX_prolong = zoomZ*NZ,zoomY*NY,zoomX*NX

        #generate old mesh 
        LZ,LY,LX = getMeshLength(config,patient.NZ,patient.NY,patient.NX)
        meshInfo_old = opticalFlow.MeshInfo3D(patient.NZ,patient.NY,patient.NX,LZ,LY,LX)

        #generate new mesh for prolongation 
        meshInfo_new = opticalFlow.MeshInfo3D(NZ_prolong,NY_prolong,NX_prolong,LZ_prolong,LY_prolong,LX_prolong)
        prolongationOp = opticalFlow.Prolongation3D(meshInfo_old,meshInfo_new,InterpolationTypeCuda,BoundaryTypeCuda)

        #convert 4d file to pytorch tensor 
        data_4d = torch.from_numpy(patient.nii_data_zyxt).float().to(DEVICE)

        #convert masks to pytorch tensor
        mask_diastole = torch.from_numpy(patient.nii_mask_diastole).float().to(DEVICE)
        mask_systole = torch.from_numpy(patient.nii_mask_systole).float().to(DEVICE)

        #prolongate 
        prolongation_diastole = prolongationOp.forward(mask_diastole)
        prolongation_systole = prolongationOp.forward(mask_systole)
        prolongation_4d = prolongationOp.forwardVectorField(data_4d.contiguous())

        # save as nifty
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
        save_torch_to_nifty(prolongation_4d, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt, zooms=(zoomX*patient.NX/NX_prolong, zoomY*patient.NY/NY_prolong,zoomZ*patient.NZ/NZ_prolong, zoomT) )
        save_torch_to_nifty(prolongation_diastole, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole, zooms=(zoomX*patient.NX/NX_prolong, zoomY*patient.NY/NY_prolong,zoomZ*patient.NZ/NZ_prolong) )
        save_torch_to_nifty(prolongation_systole, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole, zooms=(zoomX*patient.NX/NX_prolong, zoomY*patient.NY/NY_prolong,zoomZ*patient.NZ/NZ_prolong) )

        #
        xprolongfac[index] = NX_prolong/patient.NX
        yprolongfac[index] = NY_prolong/patient.NY
        zprolongfac[index] = NZ_prolong/patient.NZ

        pbar.update(1)


    # save data base
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = dataSet.df.copy()
    output_df['xprolongfac'] = xprolongfac
    output_df['yprolongfac'] = yprolongfac
    output_df['zprolongfac'] = zprolongfac
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file,index=False)

    # for index, row in df.iterrows():

    #     # Load 4D nifty [x,y,z,t]
    #     print("=======================================")
    #     PATIENT_NAME = row['Name']
    #     print("load data for patient: ", PATIENT_NAME)
    #     vol = nib.load(os.path.sep.join([VOLUMES_PATH, PATIENT_NAME + ".nii.gz"]))
    #     vol_hdr = vol.header
    #     vol_affine = vol.affine
    #     nii_data_xyzt = vol.get_fdata()
    #     NX = nii_data_xyzt.shape[0]
    #     NY = nii_data_xyzt.shape[1]
    #     NZ = nii_data_xyzt.shape[2]
    #     NT = nii_data_xyzt.shape[3]
    #     print("   * (NX,NY,NZ,NT) = ", NX, NY, NZ, NT )
    #     nii_data = np.swapaxes(nii_data_xyzt, 0, 2)
    #     data_4d = torch.from_numpy(nii_data).float().to(DEVICE)
    #     #
    #     saveDirPatient = plots.createSubDirectory(saveDirSegmentations, PATIENT_NAME)
    #     #read time steps for diastole and systole
    #     tDiastole = row["Diastole"]
    #     tSystole = row["Systole"]
    #     print("   * systole at time:  ", tSystole)
    #     print("   * diastole at time: ", tDiastole)
    #     print("=======================================")


    #     #read zooms from nifty file
    #     zooms = vol_hdr.get_zooms()
    #     # zoomX = round(zooms[0])
    #     # zoomY = round(zooms[1])
    #     # zoomZ = round(zooms[2])
    #     zoomX = zooms[0]
    #     zoomY = zooms[1]
    #     zoomZ = zooms[2]
    #     zoomT = zooms[3]
    #     # NZ_prolong,NY_prolong,NX_prolong = zoomZ*NZ,zoomY*NY,zoomX*NX

    #     #generate old mesh 
    #     LZ,LY,LX = getMeshLength(config,NZ,NY,NX)
    #     meshInfo_old = opticalFlow.MeshInfo3D(NZ,NY,NX,LZ,LY,LX)

    #     #generate new mesh for prolongation 
    #     meshInfo_new = opticalFlow.MeshInfo3D(NZ_prolong,NY_prolong,NX_prolong,LZ_prolong,LY_prolong,LX_prolong)
    #     prolongationOp = opticalFlow.Prolongation3D(meshInfo_old,meshInfo_new,InterpolationTypeCuda,BoundaryTypeCuda)

    #     # get input masks for diastole
    #     nii_mask_diastole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Diastole_Labelmap.nii"]))
    #     hdr_mask_diastole = nii_mask_diastole_load.header
    #     affine_mask_diastole = nii_mask_diastole_load.affine
    #     nii_mask_diastole_xyz = nii_mask_diastole_load.get_fdata()
    #     nii_mask_diastole = np.swapaxes(nii_mask_diastole_xyz, 0, 2)
    #     mask_diastole = torch.from_numpy(nii_mask_diastole).float().to(DEVICE)

    #     # get input masks for systole 
    #     nii_mask_systole_load = nib.load(os.path.sep.join([SEGMENTATIONS_PATH, PATIENT_NAME, PATIENT_NAME + "_Systole_Labelmap.nii"]))
    #     hdr_mask_systole = nii_mask_systole_load.header
    #     affine_mask_systole = nii_mask_systole_load.affine
    #     nii_mask_systole_xyz = nii_mask_systole_load.get_fdata()
    #     nii_mask_systole = np.swapaxes(nii_mask_systole_xyz, 0, 2)
    #     mask_systole = torch.from_numpy(nii_mask_systole).float().to(DEVICE)

    #     #prolongate 
    #     prolongation_diastole = prolongationOp.forward(mask_diastole)
    #     prolongation_systole = prolongationOp.forward(mask_systole)
    #     prolongation_4d = prolongationOp.forwardVectorField(data_4d.contiguous())

    #     # save as nifty
    #     save_torch_to_nifty(prolongation_4d, saveDir4D, PATIENT_NAME + ".nii.gz", vol_hdr, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong, zoomT) )
    #     save_torch_to_nifty(prolongation_diastole, saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii", hdr_mask_diastole, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
    #     save_torch_to_nifty(prolongation_systole, saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii", hdr_mask_systole, zooms=(zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )

    #     #
    #     xprolongfac[index] = NX_prolong/NX
    #     yprolongfac[index] = NY_prolong/NY
    #     zprolongfac[index] = NZ_prolong/NZ

    #     # #convert
    #     # prolongation_4d_np = prolongation_4d.cpu().detach().numpy()
    #     # prolongation_4d_xyzt = np.swapaxes(prolongation_4d_np, 0, 2)
    #     # #header 
    #     # ni_img_4d_hdr = nib.nifti1.Nifti1Header()
    #     # ni_img_4d_hdr.set_data_shape((NX_prolong,NY_prolong,NZ_prolong,NT))
    #     # ni_img_4d_hdr.set_qform( vol_hdr.get_qform() )
    #     # ni_img_4d_hdr.set_sform( vol_hdr.get_sform() )
    #     # ni_img_4d_hdr.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong, zoomT)  )
    #     # #img
    #     # ni_img_4d = nib.Nifti1Image(prolongation_4d_xyzt, affine=None, header=ni_img_4d_hdr)
    #     # #save
    #     # outputFile_4d = os.path.sep.join([saveDir4D, PATIENT_NAME + ".nii.gz"])
    #     # nib.save(ni_img_4d, outputFile_4d)

    #     # #save diastole as nifty
    #     # #convert 
    #     # prolongation_diastole_np = prolongation_diastole.cpu().detach().numpy()
    #     # prolongation_diastole_xyzt = np.swapaxes(prolongation_diastole_np, 0, 2)
    #     # #header 
    #     # hdr_mask_diastole_prolong = nib.nifti1.Nifti1Header()
    #     # hdr_mask_diastole_prolong.set_data_shape((NX_prolong,NY_prolong,NZ_prolong))
    #     # hdr_mask_diastole_prolong.set_qform( hdr_mask_diastole.get_qform() )
    #     # hdr_mask_diastole_prolong.set_sform( hdr_mask_diastole.get_sform() )
    #     # hdr_mask_diastole_prolong.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
    #     # #img
    #     # ni_img_diastole = nib.Nifti1Image(prolongation_diastole_xyzt, affine=None, header=hdr_mask_diastole_prolong)
    #     # #save
    #     # outputFile_diastole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Diastole_Labelmap.nii"])
    #     # nib.save(ni_img_diastole, outputFile_diastole)

    #     # #save systole as nifty
    #     # #convert 
    #     # prolongation_systole_np = prolongation_systole.cpu().detach().numpy()
    #     # prolongation_systole_xyzt = np.swapaxes(prolongation_systole_np, 0, 2)
    #     # #header 
    #     # hdr_mask_systole_prolong = nib.nifti1.Nifti1Header()
    #     # hdr_mask_systole_prolong.set_data_shape((NX_prolong,NY_prolong,NZ_prolong))
    #     # hdr_mask_systole_prolong.set_qform( hdr_mask_systole.get_qform() )
    #     # hdr_mask_systole_prolong.set_sform( hdr_mask_systole.get_sform() )
    #     # hdr_mask_systole_prolong.set_zooms( (zoomX*NX/NX_prolong, zoomY*NY/NY_prolong,zoomZ*NZ/NZ_prolong) )
    #     # #img
    #     # ni_img_systole = nib.Nifti1Image(prolongation_systole_xyzt, affine=None, header=hdr_mask_systole_prolong)
    #     # #save
    #     # outputFile_systole = os.path.sep.join([saveDirPatient, PATIENT_NAME + "_Systole_Labelmap.nii"])
    #     # nib.save(ni_img_systole, outputFile_systole)