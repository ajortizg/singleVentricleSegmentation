
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
import utils.transforms as T
from dataset import singleVentricleDataset

from opticalFlow_cuda_ext import opticalFlow


def save_torch_to_nifty(file, saveDir, fileName, hdr_old, zooms="old"):
    # convert
    file_np = file.cpu().detach().numpy()
    file_xyzt = np.swapaxes(file_np, 0, 2)
    # header
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    if zooms == "old":
        hdr.set_zooms(hdr_old.get_zooms())
    else:
        hdr.set_zooms(zooms)
    # img
    ni_img = nib.Nifti1Image(file_xyzt, affine=None, header=hdr)
    # save
    outputFile = os.path.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)
    # print("old header:")
    # print(hdr_old)
    # print("new header:")
    # print(hdr)


def getMeshLength(config, NZ, NY, NX):
    LenghtType = config.get('PROLONGATION', 'LenghtType')
    if LenghtType == "numDofs":
        LZ = NZ - 1
        LY = NY - 1
        LX = NX - 1
        return LZ, LY, LX
    elif LenghtType == "fixed":
        LZ = config.getfloat('PROLONGATION', "LenghtZ")
        LY = config.getfloat('PROLONGATION', "LenghtY")
        LX = config.getfloat('PROLONGATION', "LenghtX")
        return LZ, LY, LX


if __name__ == "__main__":

    plots.printConsoleOutput_Header("preprocessing data: prolongation")

    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')
    cuda_availabe = config.get('DEVICE', 'cuda_availabe')
    DEVICE = "cuda" if cuda_availabe else "cpu"

    # interpolation
    interType = config.get('PROLONGATION', 'InterpolationType')
    InterpolationTypeCuda = None
    if interType == "NEAREST":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
    elif interType == "LINEAR":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
    elif interType == "CUBIC_HERMITESPLINE":
        InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
    else:
        raise Exception("wrong InterpolationType in configParser")
    # boundary
    boundaryType = config.get('PROLONGATION', 'BoundaryType')
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
    LZ_prolong, LY_prolong, LX_prolong = getMeshLength(config, NZ_prolong, NY_prolong, NX_prolong)

    use_th = config.getboolean('PROLONGATION', 'USE_TH')
    bin_th = config.getfloat('PROLONGATION', 'BIN_TH')
    time_pad = config.getint('PROLONGATION', 'PAD_TIME')
    time_padder = T.PadTime(maxt=time_pad)

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_prolongation")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # load data base
    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    #
    saveDir4D = plots.createSubDirectory(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dataSet.segmentations_subdir_path)

    # generate columns for prolongation factors
    xprolongfac = np.zeros(len(dataSet))
    yprolongfac = np.zeros(len(dataSet))
    zprolongfac = np.zeros(len(dataSet))
    timeprolongfac = np.zeros(len(dataSet))
    original_NT = np.zeros(len(dataSet))

    # iterate over all patients
    pbar = tqdm(total=len(dataSet))
    for index in range(0, len(dataSet)):

        patient = dataSet[index]

        pbar.set_postfix_str(f'P: {patient.name}')

        # read zooms from nifty file
        zooms = patient.nii_header_xyzt.get_zooms()
        # zoomX = round(zooms[0])
        # zoomY = round(zooms[1])
        # zoomZ = round(zooms[2])
        zoomX = zooms[0]
        zoomY = zooms[1]
        zoomZ = zooms[2]
        zoomT = zooms[3]
        # NZ_prolong,NY_prolong,NX_prolong = zoomZ*NZ,zoomY*NY,zoomX*NX

        # generate old mesh
        LZ, LY, LX = getMeshLength(config, patient.NZ, patient.NY, patient.NX)
        meshInfo_old = opticalFlow.MeshInfo3D(patient.NZ, patient.NY, patient.NX, LZ, LY, LX)

        # generate new mesh for prolongation
        meshInfo_new = opticalFlow.MeshInfo3D(NZ_prolong, NY_prolong, NX_prolong, LZ_prolong, LY_prolong, LX_prolong)
        prolongationOp = opticalFlow.Prolongation3D(meshInfo_old, meshInfo_new, InterpolationTypeCuda, BoundaryTypeCuda)

        # convert 4d file to pytorch tensor
        data_4d = torch.from_numpy(patient.nii_data_zyxt).float().to(DEVICE)

        # convert masks to pytorch tensor
        mask_diastole = torch.from_numpy(patient.nii_mask_diastole).float().to(DEVICE)
        mask_systole = torch.from_numpy(patient.nii_mask_systole).float().to(DEVICE)

        # prolongate
        prolongation_diastole = prolongationOp.forward(mask_diastole)
        prolongation_systole = prolongationOp.forward(mask_systole)
        prolongation_4d = prolongationOp.forwardVectorField(data_4d.contiguous())

        original_NT[index] = prolongation_4d.shape[3]
        
        # Time padding
        prolongation_4d = time_padder(prolongation_4d)

        # binarize prolonganted masks
        if use_th:
            prolongation_diastole = torch.where(prolongation_diastole > bin_th, 1.0, 0.0)
            prolongation_systole = torch.where(prolongation_systole > bin_th, 1.0, 0.0)

        # save as nifty
        saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)
        save_torch_to_nifty(
            prolongation_4d, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt,
            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong, zoomT / time_pad))
        save_torch_to_nifty(prolongation_diastole, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole,
                            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong))
        save_torch_to_nifty(prolongation_systole, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole,
                            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong))

        #
        xprolongfac[index] = NX_prolong / patient.NX
        yprolongfac[index] = NY_prolong / patient.NY
        zprolongfac[index] = NZ_prolong / patient.NZ
        timeprolongfac[index] = time_pad / patient.NT

        pbar.update(1)

    # save data base
    print("\n")
    print("==================================")
    print("save database to excel file")
    output_df = dataSet.df.copy()
    output_df['xprolongfac'] = xprolongfac
    output_df['yprolongfac'] = yprolongfac
    output_df['zprolongfac'] = zprolongfac
    output_df['timeprolongfac'] = timeprolongfac
    output_df['original_NT'] = original_NT
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
