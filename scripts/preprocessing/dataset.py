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

from utilities import path_utils
from utilities.plots import plots


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

    def img_array(self) -> np.ndarray:
        """
        Returns the 4D image data as a NumPy array.
        """
        return self.img.get_fdata() if self.img else np.array([])

    def seg_dia_array(self) -> np.ndarray:
        """
        Returns the diastole segmentation mask data as a NumPy array.
        """
        return self.seg_dia.get_fdata() if self.seg_dia else np.array([])

    def seg_sys_array(self) -> np.ndarray:
        """
        Returns the systole segmentation mask data as a NumPy array.
        """
        return self.seg_sys.get_fdata() if self.seg_sys else np.array([])

    def img_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool):
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
        self.img = nib.Nifti1Image(x, affine=affine, header=header)

    def seg_dia_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool):
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
        self.seg_dia = nib.Nifti1Image(x, affine=affine, header=header)

    def seg_sys_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool):
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
        self.seg_sys = nib.Nifti1Image(x, affine=affine, header=header)

    def save_nifti(self, out_img_dir, out_seg_dir):
        """
        Saves the Nifti images for the 4D image and segmentation masks to disk.

        Args:
            out_img_dir (str): Directory to save the 4D image.
            out_seg_dir (str): Directory to save the segmentation masks.
        """
        out_patient_dir = path_utils.create_sub_dir(out_seg_dir, self.name)
        nib.save(self.img, osp.join(out_img_dir, self.name + ".nii.gz"))
        nib.save(self.seg_dia, osp.join(out_patient_dir, self.name + "_Diastole_Labelmap.nii.gz"))
        nib.save(self.seg_sys, osp.join(out_patient_dir, self.name + "_Systole_Labelmap.nii.gz"))

    def viz_data(self, save_dir, gif, dur):
        save_dir = path_utils.create_sub_dir(save_dir, self.name)
        img = self.img_array().transpose((3, 2, 1, 0))  # x,y,z,t -> t,z,y,x
        for t in range(img.shape[0]):
            if t == self.tsys:
                seg = self.seg_sys_array().swapaxes(2, 0)
            elif t == self.tdia:
                seg = self.seg_dia_array().swapaxes(2, 0)
            else:
                seg = None
            plots.save_overlaped_img_mask_numpy(img[t], seg, '{}_{}.png'.format(self.name, t), save_dir, 0.3)

        if gif:
            frames = [Image.open(image) for image in natsorted(glob.glob(f"{save_dir}/*.png"))]
            frame_one = frames[0]
            frame_one.save(osp.join(save_dir, "animation.gif"), format="GIF", append_images=frames,
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
        seg_dia = nib.load(osp.join(self.segs_dir, name, name + "_Diastole_Labelmap.nii.gz"))
        seg_sys = nib.load(osp.join(self.segs_dir, name, name + "_Systole_Labelmap.nii.gz"))

        return Patient(name, tsys, tdia,  min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)
