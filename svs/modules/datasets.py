import os
import numpy as np
import nibabel as nib
import os.path as osp
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from dataclasses import dataclass
from PIL import Image
import glob
from natsort import natsorted

from svs.utils import dirs
from svs.utils import plots


xyzt_to_zyxt = (2, 1, 0, 3)
zyxt_to_xyzt = (2, 1, 0, 3)
xyz_to_zyx = (2, 1, 0)
zyx_to_xyz = (2, 1, 0)
xyzt_to_tzyx = (3, 2, 1, 0)


@dataclass
class Patient:
    """
    Data class representing a patient with MRI and segmentation data.
    """
    name: str = ""
    tsys: int = 0
    tdia: int = 0
    init_ts: int = 0
    final_ts: int = 0
    img: Optional[nib.Nifti1Image] = None
    seg_dia: Optional[nib.Nifti1Image] = None
    seg_sys: Optional[nib.Nifti1Image] = None

    def get_img_array(self) -> np.ndarray:
        """
        Returns the 4D image data as a NumPy array with shape [x,y,z,t]
        """
        return self.img.get_fdata() if self.img else np.array([])

    def get_seg_dia_array(self) -> np.ndarray:
        """
        Returns the diastole segmentation mask data as a NumPy array with shape [x,y,z]
        """
        return self.seg_dia.get_fdata() if self.seg_dia else np.array([])

    def get_seg_sys_array(self) -> np.ndarray:
        """
        Returns the systole segmentation mask data as a NumPy array with shape [x,y,z]
        """
        return self.seg_sys.get_fdata() if self.seg_sys else np.array([])

    def set_img_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
        """
        Creates a Nifti1Image from a NumPy array for the 4D image data.

        Args:
            x (np.ndarray): The image data array.
            affine (np.ndarray): The affine transformation for the image.
            header (nib.Nifti1Header): The header for the Nifti image.
            update_shape (bool): Whether to update the header shape to match the array.
        """
        if update_shape:
            header.set_data_shape(x.shape)
        if zooms is not None:
            header.set_zooms(zooms)
        self.img = nib.Nifti1Image(x, affine=affine, header=header)

    def set_seg_dia_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
        """
        Creates a Nifti1Image from a NumPy array for the diastole segmentation mask.

        Args:
            x (np.ndarray): The segmentation data array.
            affine (np.ndarray): The affine transformation for the segmentation.
            header (nib.Nifti1Header): The header for the Nifti segmentation.
            update_shape (bool): Whether to update the header shape to match the array.
        """
        if update_shape:
            header.set_data_shape(x.shape)
        if zooms is not None:
            header.set_zooms(zooms)
        self.seg_dia = nib.Nifti1Image(x, affine=affine, header=header)

    def set_seg_sys_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
        """
        Creates a Nifti1Image from a NumPy array for the systole segmentation mask.

        Args:
            x (np.ndarray): The segmentation data array.
            affine (np.ndarray): The affine transformation for the segmentation.
            header (nib.Nifti1Header): The header for the Nifti segmentation.
            update_shape (bool): Whether to update the header shape to match the array.
        """
        if update_shape:
            header.set_data_shape(x.shape)
        if zooms is not None:
            header.set_zooms(zooms)
        self.seg_sys = nib.Nifti1Image(x, affine=affine, header=header)

    def save_nifti(self, out_img_dir, out_seg_dir):
        """
        Saves the Nifti images for the 4D image and segmentation masks to disk.

        Args:
            out_img_dir (str): Directory to save the 4D image.
            out_seg_dir (str): Directory to save the segmentation masks.
        """
        out_patient_dir = dirs.create_subdir(out_seg_dir, self.name)
        nib.save(self.img, osp.join(out_img_dir, f'{self.name}.nii.gz'))
        nib.save(self.seg_dia, osp.join(out_patient_dir,  f"{self.name}_Diastole_Labelmap.nii.gz"))
        nib.save(self.seg_sys, osp.join(out_patient_dir, f"{self.name}_Systole_Labelmap.nii.gz"))

    def viz_data(self, save_dir, gif, dur, aspect_ratio=1.7):
        """
        Visualizes the data and saves it as images or a GIF.

        Args:
            save_dir (str): Directory to save the visualizations.
            gif (bool): Whether to create a GIF.
            dur (int): Duration of each frame in the GIF (in milliseconds).
            aspect_ratio (float, optional): Aspect ratio for visualization. Defaults to 1.7.
        """
        save_dir = dirs.create_subdir(save_dir, self.name)
        img = self.get_img_array().transpose(xyzt_to_tzyx) 
        for t in range(img.shape[0]):
            if t == self.tsys:
                seg = self.get_seg_sys_array().swapaxes(2, 0)
            elif t == self.tdia:
                seg = self.get_seg_dia_array().swapaxes(2, 0)
            else:
                seg = None
            plots.save_image_mask_overlay(img[t], seg, f'{self.name}_{t}.png', save_dir, 0.3, aspect_ratio)

        if gif:
            frames = [Image.open(image) for image in natsorted(glob.glob(f'{save_dir}/*.png'))]
            frame_one = frames[0]
            frame_one.save(osp.join(save_dir, 'animation.gif'), format='GIF', append_images=frames,
                           save_all=True, duration=dur, loop=0)


class MRIBaseDataset(Dataset):
    def __init__(self, base_dir, imgs_dir, segs_dir, metadata_file):
        """
        Initializes the MRIBaseDataset instance.

        Args:
            base_dir (str): Base directory containing the dataset.
            imgs_dir (str): Sub-directory containing the images.
            segs_dir (str): Sub-directory containing the segmentations.
            metadata_file (str): File name of the metadata Excel file.
        """
        self.imgs_dir = osp.join(base_dir, imgs_dir)
        self.segs_dir = osp.join(base_dir, segs_dir)
        self.df = pd.read_excel(osp.join(base_dir, metadata_file))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Patient:
        row = self.df.iloc[idx]
        name = row["Name"]
        tsys = row["Systole"]
        tdia = row["Diastole"]

        # Load 4D image [x,y,z,t]
        img = nib.load(osp.join(self.imgs_dir, f'{name}.nii.gz'))

        # Load segmentation masks [x,y,z]
        seg_dia = nib.load(osp.join(self.segs_dir, name, f'{name}_Diastole_Labelmap.nii.gz'))
        seg_sys = nib.load(osp.join(self.segs_dir, name,   f'{name}_Systole_Labelmap.nii.gz'))

        return Patient(name, tsys, tdia,  min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)


class RawACDCDadataset(Dataset):
    def __init__(self, base_dir):
        """
        Initializes the dataset by listing all patient directories.

        Args:
            base_dir (str): The base directory containing patient data.
        """
        self.data_dirs = sorted([osp.join(base_dir, d) for d in os.listdir(base_dir) if osp.isdir(osp.join(base_dir, d))])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx) -> Patient:
        """
        Retrieves a patient data entry including image and segmentation masks.

        Args:
            idx (int): Index of the patient directory to load.

        Returns:
            Patient: A dataclass containing the patient's data.
        """
        patient_dir = self.data_dirs[idx]
        name = osp.basename(patient_dir)
        tdia, tsys = self.extract_diastole_systole_times(patient_dir)

        # Load 4D image [x,y,z,t]
        img = nib.load(osp.join(patient_dir, f'{name}_4d.nii.gz'))

        # Load segmentation masks [x,y,z]
        seg_dia = nib.load(osp.join(patient_dir,  f'{name}_frame{tdia:02d}_gt.nii.gz'))
        seg_sys = nib.load(osp.join(patient_dir, f'{name}_frame{tsys:02d}_gt.nii.gz'))

        return Patient(name, tsys, tdia,  min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)

    def extract_diastole_systole_times(self, data_dir):
        """
        Extracts the diastole and systole times from the Info.cfg file.

        Args:
            data_dir (str): The directory containing the Info.cfg file.

        Returns:
            Tuple[int, int]: The diastole and systole times.
        """
        with open(osp.join(data_dir, 'Info.cfg'), 'r') as f:
            tdia = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            tsys = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
        return tdia, tsys
