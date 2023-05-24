import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import configparser
import os.path as osp

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import plots
from dataset.singleVentricleDataset import SingleVentricleDataset
from dataset.preprocessing.preprocessing_cutting import cut_patient
from dataset.preprocessing.preprocessing_prolongation import prolongate_patient
from dataset.preprocessing.preprocessing_normalization import normalize_patient
from dataset.preprocessing.preprocessing_flipping import flip_patient


def save_np_to_nifty(file, saveDir, fileName, hdr_old):
    # header
    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_shape(file.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    hdr.set_zooms(hdr_old.get_zooms())
    # img
    ni_img = nib.Nifti1Image(file, affine=None, header=hdr)
    # save
    outputFile = osp.sep.join([saveDir, fileName])
    nib.save(ni_img, outputFile)


if __name__ == "__main__":
    # load config parser
    config = configparser.ConfigParser()
    config.read('parser/configPreprocessing.ini')

    # create save directory
    saveDir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "preprocessing")

    # save config file to save directory
    conifgOutput = os.path.sep.join([saveDir, "config.ini"])
    with open(conifgOutput, 'w') as configfile:
        config.write(configfile)

    dset = SingleVentricleDataset(config)
    name = config.get('DATA', 'PATIENT_NAME')
    try:
        index = dset.index_for_patient(name)
        patient = dset[index]
    except IndexError:
        print(name + ' not found!')
        sys.exit()

    img4d = patient.nii_data_xyzt
    md = patient.nii_mask_diastole_xyz
    ms = patient.nii_mask_systole_xyz
    df = dset.df

    # print('Flipping ...')
    # md, ms, df = flip_patient(config, patient.name, md, ms, df)

    print('Cutting ...')
    img4d, md, ms, df = cut_patient(config, img4d, md, ms, df)

    print('Prolongation ...')
    img4d, md, ms, df = prolongate_patient(config, img4d, md, ms, df)

    print('Normalization ...')
    img4d = normalize_patient(config, img4d, md, ms, patient.tDiastole, patient.tSystole)

    # print(img4d.shape, md.shape, ms.shape)
    # print(df)

    # save paths
    saveDir4D = plots.createSubDirectory(saveDir, dset.volumes_subdir_path)
    saveDirSegmentations = plots.createSubDirectory(saveDir, dset.segmentations_subdir_path)
    saveDirPatient = plots.createSubDirectory(saveDirSegmentations, patient.name)

    save_np_to_nifty(img4d, saveDir4D, patient.name + ".nii.gz", patient.nii_header_xyzt)
    save_np_to_nifty(md, saveDirPatient, patient.name + "_Diastole_Labelmap.nii", patient.hdr_mask_diastole)
    save_np_to_nifty(ms, saveDirPatient, patient.name + "_Systole_Labelmap.nii", patient.hdr_mask_systole)

    output_df_file = os.path.sep.join([saveDir, dset.segmentations_filename])
    df.to_excel(output_df_file, index=False)
