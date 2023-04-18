import os
import os.path as osp
import sys
import shutil

from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import nibabel as nib


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import file_paths_utils as fpu
from thirdparty.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def save_np_to_nifty(x, save_dir, filename, affine, hdr_old):
    hdr = nib.nifti1.Nifti1Header.from_header(hdr_old)
    hdr.set_data_shape(x.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    hdr.set_zooms(hdr_old.get_zooms()[:-1])
    nii_data = nib.Nifti1Image(x, affine=affine, header=hdr)
    nib.save(nii_data, osp.join(save_dir, filename))


if __name__ == '__main__':
    # Source paths
    root_dir = 'data/singleVentricleData'
    imgs_dir = osp.join(root_dir, 'NIFTI_4D_Datasets')
    labels_dir = osp.join(root_dir, 'NIFTI_Single_Ventricle_Segmentations')

    # Split in training and testing
    df = pd.read_excel(osp.join(root_dir, 'Segmentation_volumes.xlsx'))
    df_train = df[(df['Split'] == 'train')]
    df_train.reset_index(inplace=True)
    df_test = df[df['Split'] == 'test']
    df_test.reset_index(inplace=True)

    # Output paths
    save_dir = fpu.create_save_dir('results', 'svd_nnunet_raw')
    train_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTr')
    train_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTr')
    test_imgs_dir = fpu.create_sub_dir(save_dir, 'imagesTs')
    test_labels_dir = fpu.create_sub_dir(save_dir, 'labelsTs')

    k = 1
    for i in tqdm(range(len(df_train))):
        df_row = df_train.iloc[[i]]
        patient_name = df_row.loc[i, 'Name']
        es = df_row.loc[i, 'Systole']
        ed = df_row.loc[i, 'Diastole']

        times = [ed, es]
        endings = ['_Diastole_Labelmap.nii', '_Systole_Labelmap.nii']
        nii_img = nib.load(osp.join(imgs_dir, patient_name + '.nii.gz'))

        for t, ending in zip(times, endings):
            img = nii_img.get_fdata()[..., t]
            label = nib.load(osp.join(labels_dir, patient_name, patient_name + ending))

            save_np_to_nifty(img, train_imgs_dir, 'SVD_{:03d}_0001.nii.gz'.format(k), nii_img.affine, nii_img.header)
            nib.save(label, osp.join(train_labels_dir, 'SVD_{:03d}.nii.gz'.format(k)))
            k += 1

    generate_dataset_json(output_folder=save_dir,
                          channel_names={1: 'T1'},
                          labels={'background': 0, 'SV': 1},
                          num_training_cases=k - 1,
                          file_ending='.nii.gz',
                          dataset_name='SVD',
                          overwrite_image_reader_writer='NibabelIOWithReorient')
