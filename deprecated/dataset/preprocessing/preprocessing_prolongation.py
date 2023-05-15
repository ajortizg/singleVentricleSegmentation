
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import pandas
import configparser
import torch
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../../'))
sys.path.append(ROOT_DIR)
# from utilities import plots
from utilities import path_utils
from deprecated.dataset import singleVentricleDataset
import utilities.transforms.unary_transforms as T1

from opticalFlow_cuda_ext import opticalFlow

__all__ = ['prolongate_patient']


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


# def prolongate_patient(config, img4d, md, ms, df):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # interpolation
#     interType = config.get('PROLONGATION', 'InterpolationType')
#     InterpolationTypeCuda = None
#     if interType == "NEAREST":
#         InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
#     elif interType == "LINEAR":
#         InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
#     elif interType == "CUBIC_HERMITESPLINE":
#         InterpolationTypeCuda = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
#     else:
#         raise Exception("wrong InterpolationType in configParser")
#     # boundary
#     boundaryType = config.get('PROLONGATION', 'BoundaryType')
#     BoundaryTypeCuda = None
#     if boundaryType == "NEAREST":
#         BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_NEAREST
#     elif boundaryType == "MIRROR":
#         BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_MIRROR
#     elif boundaryType == "REFLECT":
#         BoundaryTypeCuda = opticalFlow.BoundaryType.BOUNDARY_REFLECT
#     else:
#         raise Exception("wrong BoundaryType in configParser")

#     # prolongation size
#     NX_prolong = config.getint('PROLONGATION', 'NX_prolong')
#     NY_prolong = config.getint('PROLONGATION', 'NY_prolong')
#     NZ_prolong = config.getint('PROLONGATION', 'NZ_prolong')
#     LZ_prolong, LY_prolong, LX_prolong = getMeshLength(config, NZ_prolong, NY_prolong, NX_prolong)

#     use_th = config.getboolean('PROLONGATION', 'USE_TH')
#     bin_th = config.getfloat('PROLONGATION', 'BIN_TH')
#     time_pad = config.getint('PROLONGATION', 'PAD_TIME')
#     time_padder = T1.PadTime(maxt=time_pad)

#     # xprolongfac = np.zeros(len(dataSet))
#     # yprolongfac = np.zeros(len(dataSet))
#     # zprolongfac = np.zeros(len(dataSet))

#     # generate old mesh
#     NX, NY, NZ, NT = img4d.shape
#     LZ, LY, LX = getMeshLength(config, NZ, NY, NX)
#     meshInfo_old = opticalFlow.MeshInfo3D(NZ, NY, NX, LZ, LY, LX)

#     # generate new mesh for prolongation
#     meshInfo_new = opticalFlow.MeshInfo3D(NZ_prolong, NY_prolong, NX_prolong, LZ_prolong, LY_prolong, LX_prolong)
#     prolongationOp = opticalFlow.Prolongation3D(meshInfo_old, meshInfo_new, InterpolationTypeCuda, BoundaryTypeCuda)

#     # convert 4d file to pytorch tensor
#     img4d_zyxt = np.swapaxes(img4d, 0, 2)
#     img4d_zyxt = torch.from_numpy(img4d_zyxt).float().to(device)

#     # convert masks to pytorch tensor
#     md_zyx = np.swapaxes(md, 0, 2)
#     ms_zyx = np.swapaxes(ms, 0, 2)
#     md_zyx = torch.from_numpy(md_zyx).float().to(device)
#     ms_zyx = torch.from_numpy(ms_zyx).float().to(device)

#     # prolongate
#     prolongation_diastole = prolongationOp.forward(md_zyx)
#     prolongation_systole = prolongationOp.forward(ms_zyx)
#     prolongation_4d = prolongationOp.forwardVectorField(img4d_zyxt.contiguous())

#     # Time padding
#     prolongation_4d = time_padder(prolongation_4d)

#     # binarize prolonganted masks
#     if use_th:
#         prolongation_diastole = torch.where(prolongation_diastole > bin_th, 1.0, 0.0)
#         prolongation_systole = torch.where(prolongation_systole > bin_th, 1.0, 0.0)

#     img4d = np.swapaxes(prolongation_4d.cpu().detach().numpy(), 0, 2)
#     md = np.swapaxes(prolongation_diastole.cpu().detach().numpy(), 0, 2)
#     ms = np.swapaxes(prolongation_systole.cpu().detach().numpy(), 0, 2)

#     output_df = df.copy()
#     output_df['xprolongfac'] = NX_prolong / NX
#     output_df['yprolongfac'] = NY_prolong / NY
#     output_df['zprolongfac'] = NZ_prolong / NZ
#     output_df['timeprolongfac'] = time_pad / patient.NT

#     return img4d, md, ms, output_df


if __name__ == "__main__":
    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/deprecated/configPreprocessing.ini')
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

    # use_th = config.getboolean('PROLONGATION', 'USE_TH')
    # bin_th = config.getfloat('PROLONGATION', 'BIN_TH')
    time_pad = config.getint('PROLONGATION', 'PAD_TIME')
    time_padder = T1.PadTime(maxt=time_pad)

    # create save directory
    saveDir = path_utils.create_save_dir(config.get('DATA', 'OUTPUT_PATH'), "preprocessing_prolongation")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    # load data base
    dataSet = singleVentricleDataset.SingleVentricleDataset(config)

    #
    saveDir4D = path_utils.create_sub_dir(saveDir, dataSet.volumes_subdir_path)
    saveDirSegmentations = path_utils.create_sub_dir(saveDir, dataSet.segmentations_subdir_path)

    # generate columns for prolongation factors
    xprolongfac = np.zeros(len(dataSet))
    yprolongfac = np.zeros(len(dataSet))
    zprolongfac = np.zeros(len(dataSet))
    timeprolongfac = np.zeros(len(dataSet))

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

        # Time padding
        prolongation_4d = time_padder(prolongation_4d)

        # binarize prolonganted masks
        # if use_th:
        # prolongation_diastole = torch.where(prolongation_diastole > bin_th, 1.0, 0.0)
        # prolongation_systole = torch.where(prolongation_systole > bin_th, 1.0, 0.0)
        prolongation_diastole = prolongation_diastole.round()
        prolongation_systole = prolongation_systole.round()
        print(torch.unique(prolongation_diastole))
        print(torch.unique(prolongation_systole))

        # save as nifty
        saveDirPatient = path_utils.create_sub_dir(saveDirSegmentations, patient.name)
        save_torch_to_nifty(
            prolongation_4d, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt,
            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong,
                   zoomT / time_pad))
        save_torch_to_nifty(
            prolongation_diastole, saveDirPatient, patient.name + "_Diastole_Labelmap.nii.gz", patient.hdr_mask_diastole,
            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong))
        save_torch_to_nifty(
            prolongation_systole, saveDirPatient, patient.name + "_Systole_Labelmap.nii.gz", patient.hdr_mask_systole,
            zooms=(zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong))

        if patient.full_cycle:
            for t in range(patient.NT):
                mask = torch.from_numpy(patient.nii_masks_zyx[t]).float().to(DEVICE)
                mask_prolongated = prolongationOp.forward(mask)
                # if use_th:
                mask_prolongated =mask_prolongated.round()
                print(torch.unique(mask_prolongated))
                mask_filename = patient.masks_dirs[t].split('/')[-1]
                save_torch_to_nifty(mask_prolongated, saveDirPatient, mask_filename, patient.nii_masks_load[t].header, zooms=(
                    zoomX * patient.NX / NX_prolong, zoomY * patient.NY / NY_prolong, zoomZ * patient.NZ / NZ_prolong))

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
    output_df_file = os.path.sep.join([saveDir, dataSet.segmentations_filename])
    output_df.to_excel(output_df_file, index=False)
