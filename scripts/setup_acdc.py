import numpy as np
import os
import nibabel as nib
import os.path as osp
from tqdm import tqdm
import pandas as pd
import shutil
import yaml
from ml_collections import config_dict

from utilities import path_utils


class ACDCPatient:
    """
    Represents an ACDC patient with their associated imaging data.
    """

    def __init__(self, name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole):
        """
        Initializes the ACDCPatient instance.

        Args:
            name (str): Patient's name or identifier.
            img4d_nii (Nifti1Image): 4D image data.
            mask_systole_nii (Nifti1Image): Systole mask image data.
            mask_diastole_nii (Nifti1Image): Diastole mask image data.
            tsystole (int): Time frame index for systole.
            tdiastole (int): Time frame index for diastole.
        """
        self.name = name
        self.img4d_nii = img4d_nii
        self.mask_systole_nii = mask_systole_nii
        self.mask_diastole_nii = mask_diastole_nii
        self.tsystole = tsystole
        self.tdiastole = tdiastole

    def save(self, save4d_dir, save_segmentations_dir):
        """
        Saves the patient's 4D image and segmentation masks.

        Args:
            save4d_dir (str): Directory to save the 4D images.
            save_segmentations_dir (str): Directory to save the segmentation masks.

        Returns:
            pd.DataFrame: DataFrame containing patient's metadata.
        """
        # Save 4d data
        out_file = osp.sep.join([save4d_dir, self.name + ".nii.gz"])
        nib.save(self.img4d_nii, out_file)

        # Save masks
        if self.mask_systole_nii is not None and self.mask_diastole_nii is not None:
            save_patient_seg_dir = path_utils.create_sub_dir(save_segmentations_dir, self.name)
            out_file = osp.sep.join([save_patient_seg_dir, self.name + "_Systole_Labelmap.nii.gz"])
            nib.save(self.mask_systole_nii, out_file)
            out_file = osp.sep.join([save_patient_seg_dir, self.name + "_Diastole_Labelmap.nii.gz"])
            nib.save(self.mask_diastole_nii, out_file)

        # Create DataFrame for patient's metadata
        row = {'Name': self.name, 'Systole': self.tsystole, 'Diastole': self.tdiastole}
        df_patient = pd.DataFrame(row, index=[0])
        return df_patient


class ACDCDataset:
    """
    Dataset class for ACDC data.
    """

    def __init__(self, base_path, label):
        """
        Initializes the ACDCDataset instance.

        Args:
            base_path (str): Base directory containing patient data.
            label (int): Label value to extract from the segmentation masks.
        """
        self.base_path = base_path
        self.patient_dirs = sorted(os.listdir(base_path))
        self.label = label

    def __getitem__(self, idx):
        """
        Gets a patient by index.

        Args:
            idx (int): Index of the patient.

        Returns:
            ACDCPatient: The patient object.
        """
        name = self.patient_dirs[idx]
        patient_dir = osp.join(self.base_path, name)
        img4d_nii = nib.load(osp.join(patient_dir, name + '_4d.nii.gz'))

        # Read systole and diastole time frames
        with open(osp.join(patient_dir, 'Info.cfg'), 'r') as f:
            tdiastole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            tsystole = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])

        mode = osp.basename(self.base_path)
        if mode == 'training':
            mask_diastole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tdiastole)]))
            mask_systole_nii = nib.load(osp.sep.join([self.base_path, name, name + '_frame%02d_gt.nii.gz' % (tsystole)]))
            mask_diastole_nii = self.extract_label(mask_diastole_nii)
            mask_systole_nii = self.extract_label(mask_systole_nii)
        else:
            mask_diastole_nii = None
            mask_systole_nii = None

        patient = ACDCPatient(name, img4d_nii, mask_systole_nii, mask_diastole_nii, tsystole, tdiastole)
        return patient

    def extract_label(self, mask_nii):
        """
        Extracts the specified label from the mask.

        Args:
            mask_nii (Nifti1Image): Input mask image.

        Returns:
            Nifti1Image: Mask image with the specified label extracted.
        """
        mask_data = mask_nii.get_fdata()
        mask_data_label = np.where(mask_data == self.label, 1.0, 0.0)
        mask_nii = nib.Nifti1Image(mask_data_label, affine=mask_nii.affine, header=mask_nii.header)
        return mask_nii

    def __len__(self):
        """
        Returns the number of patients in the dataset.

        Returns:
            int: Number of patients.
        """
        return len(self.patient_dirs)


def main(config_dir='conf', config_name='setup_acdc.yaml'):
    """
    Main function to process the ACDC dataset based on configuration.

    Args:
        config_dir (str): Directory containing the configuration file.
        config_name (str): Configuration file name.
    """
    # background: 0
    labels = {'right-ventricle': 1,
              'myocardium': 2,
              'left-ventricle': 3}

    # Load configuration
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    for roi, tag in labels.items():
        print('Extracting:', roi)
        save_dir = path_utils.create_save_dir(cfg.save_dir, f'acdc_{roi}')
        save4d_dir = path_utils.create_sub_dir(save_dir, cfg.out_imgs_dir)
        save_segmentations_dir = path_utils.create_sub_dir(save_dir, cfg.out_segs_dir)

        # Initialize DataFrame for dataset metadata
        df_dset = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
        train_ds = ACDCDataset(cfg.raw_dir, label=tag)

        for i in tqdm(range(len(train_ds))):
            patient = train_ds[i]
            df = patient.save(save4d_dir, save_segmentations_dir)
            df_dset = pd.concat([df_dset, df], ignore_index=True)

        # Save metadata and configuration
        df_dset.to_excel(osp.join(save_dir, cfg.out_metadata_file), index=False)
        shutil.copyfile(osp.join(config_dir, config_name), osp.join(save_dir, config_name))


if __name__ == "__main__":
    main()
