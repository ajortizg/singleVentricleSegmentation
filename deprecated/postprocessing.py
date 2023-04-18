import nibabel as nib
import os.path as osp
import configparser
import sys
from glob import glob
from tqdm import tqdm
import os
import pandas as pd
import re
from scipy import ndimage
import nrrd
import numpy as np
from singleVentricleDataset import SingleVentricleDataset

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utilities import plots
import utilities.quaternary_transforms as T


def filter_dirs(dirs, base_dir):
    filtered_dirs = []
    for dir in dirs:
        path = osp.join(base_dir, dir)
        if os.path.isdir(path):
            filtered_dirs.append(path)
    return sorted(filtered_dirs)


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

def prolongation(mask_data_xyz, row, idx):
    xprolongfac = row.loc[idx, 'xprolongfac']
    yprolongfac = row.loc[idx, 'yprolongfac']
    zprolongfac = row.loc[idx, 'zprolongfac']

    mask_data_zyx = np.swapaxes(mask_data_xyz, 0, 2)
    new_z = mask_data_zyx.shape[0] / zprolongfac
    new_y = mask_data_zyx.shape[1] / yprolongfac
    new_x = mask_data_zyx.shape[2] / xprolongfac
    resize = T.Resize(size=(new_z, new_y, new_x))
    return resize(mask_data_zyx).swapaxes(0, 2) # return xyz

def flip(mask_xyz, row, idx):
    xflip = row.loc[idx, 'xflip']
    yflip = row.loc[idx, 'yflip']
    zflip = row.loc[idx, 'zflip']
    
    mask_zyx = np.swapaxes(mask_xyz, 0, 2)
    mask_flipped = None
    axis = []
    if xflip:
        axis.append(0)
    if yflip:
        axis.append(1)
    if zflip:
        axis.append(2)
    mask_flipped = np.flip(mask_zyx, axis=axis).copy()
    return np.swapaxes(mask_flipped, 0, 2)  # return xyz mask

def save_nifty(mask_xyz, save_dir, filename, header=None):
    mt_nii = nib.Nifti1Image(mask_xyz, affine=None, header=header)
    outputFile = osp.sep.join([save_dir, filename])
    nib.save(mt_nii, outputFile)

def get_metadata(dataset:SingleVentricleDataset, patient_name):
    idx = dataset.index_for_patient(patient_name)
    patient = dataset[idx]
    mask_size_xyz = patient.nii_mask_diastole_xyz.shape
    return (patient.hdr_mask_diastole, mask_size_xyz)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configPostprocessing.ini')
    
    PREPROCESSED_SEGMETATIONS_FILE = config.get('PARAMETERS', 'PREPROCESSED_SEGMETATIONS_FILE')
    CNN_OUTPUT_PATH = config.get('PARAMETERS', 'CNN_OUTPUT_PATH')
    FWD_DIR = config.get('PARAMETERS', 'FWD_DIR')
    BWD_DIR = config.get('PARAMETERS', 'BWD_DIR')

    BINARY_CLOSING = config.getboolean('PARAMETERS', 'BINARY_CLOSING')
    BC_KERNEL = config.getint('PARAMETERS','BC_KERNEL')
    BINARY_OPENING = config.getboolean('PARAMETERS', 'BINARY_OPENING')
    BO_KERNEL = config.getint('PARAMETERS','BO_KERNEL')

    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), "postprocessing")

    orig_dataset = SingleVentricleDataset(config)

    patients_cnn_list = filter_dirs(os.listdir(CNN_OUTPUT_PATH), CNN_OUTPUT_PATH)
    prep_df = pd.read_excel(PREPROCESSED_SEGMETATIONS_FILE)
    pbar = tqdm(total=len(patients_cnn_list))


    for patient_path in patients_cnn_list:
        patient_name = patient_path.split(os.path.sep)[-1]
        fwd_dir = osp.join(patient_path, FWD_DIR)
        bwd_dir = osp.join(patient_path, BWD_DIR)
        fwd_masks_file = sorted(glob(osp.join(fwd_dir, '*.nii')), key=natural_keys)
        bwd_masks_file = sorted(glob(osp.join(bwd_dir, '*.nii')), key=natural_keys)

        prep_row = prep_df[prep_df['Name'] == patient_name]
        idx = prep_row.index[0]
        # xshift = prep_row.loc[idx, 'xshift']
        # yshift = prep_row.loc[idx, 'yshift']
        # zshift = prep_row.loc[idx, 'zshift']
        
        save_patient_dir = plots.createSubDirectory(save_dir, patient_name)
        save_patient_dir_fwd = plots.createSubDirectory(save_patient_dir, FWD_DIR)
        save_patient_dir_bwd = plots.createSubDirectory(save_patient_dir, BWD_DIR)

        hdr, orig_size = get_metadata(orig_dataset, patient_name)

        list_masks_files = [fwd_masks_file, bwd_masks_file]

        for list_masks in list_masks_files:
            for mask_file in list_masks:
                mask_nii = nib.load(mask_file)
                mask_data = mask_nii.get_fdata()

                if BINARY_CLOSING:
                    mask_data = ndimage.binary_closing(mask_data, structure=np.ones((BC_KERNEL,BC_KERNEL,BC_KERNEL))).astype(int)
                if BINARY_OPENING:
                    mask_data = ndimage.binary_opening(mask_data, structure=np.ones((BO_KERNEL,BO_KERNEL,BO_KERNEL))).astype(int)
                
                # mask_data = ndimage.binary_fill_holes(mask_data).astype(int)
                mask_prol = prolongation(mask_data, prep_row, idx)
                mask_flip = flip(mask_prol, prep_row, idx)
                # mask_bc = binary_closing(mask_flip)            
            
                # NX,NY,NZ = mask_flip.shape
                # diff_x = orig_size[0] - NX
                # diff_y = orig_size[1] - NY
                # diff_z = orig_size[2] - NZ

                # mask_pad = np.pad(mask_flip, ((diff_x // 2, (diff_x - diff_x // 2)),(diff_y // 2, (diff_y - diff_y // 2)),(diff_z // 2, diff_z - diff_z // 2)))
                # mask_pad = np.pad(mask_flip,((diff_x // 2,xshift),(diff_y // 2,yshift),(diff_z // 2,zshift)))
                mode = mask_file.split(osp.sep)[-2]
                filename = mask_file.split(osp.sep)[-1].split('.')[0]
                if mode == 'fwd':
                    save_nifty(mask_flip, save_patient_dir_fwd, filename + '.nii', header=hdr)
                    nrrd.write(osp.join(save_patient_dir_fwd, filename+'.nrrd'), mask_flip)
                else:
                    save_nifty(mask_flip, save_patient_dir_bwd, filename + '.nii', header=hdr)
                    nrrd.write(osp.join(save_patient_dir_bwd, filename+'.nrrd'), mask_flip)
        pbar.update(1)
       
          
    

            