from monai.data import NibabelWriter
import os
import monai.data
import monai.transforms
import monai.transforms.io
import numpy as np
import nibabel as nib
import torch
import os.path as osp
import pandas as pd
from typing import Optional
from torchvision.utils import make_grid, draw_segmentation_masks
from torch.utils.data import Dataset
from dataclasses import dataclass
from PIL import Image
import glob
from natsort import natsorted
import monai
from monai.transforms import LoadImage
from monai.data.meta_tensor import MetaTensor

from svs.utils import dirs
from svs.utils import plots
from svs.utils import io

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
    img: Optional[MetaTensor] = None        # [t,x,y,z]
    seg_dia: Optional[MetaTensor] = None    # [c,x,y,z]
    seg_sys: Optional[MetaTensor] = None    # [c,x,y,z]

    def write_nifti(self, out_img_dir, out_seg_dir):
        """
        Saves the Nifti images for the 4D image and segmentation masks to disk.

        Args:
            out_img_dir (str): Directory to save the 4D image.
            out_seg_dir (str): Directory to save the segmentation masks.
        """
        io.write_metatensor_to_nifti(self.img, osp.join(out_img_dir, f'{self.name}.nii.gz'))

        out_patient_dir = dirs.create_subdir(out_seg_dir, self.name)
        io.write_metatensor_to_nifti(self.seg_dia, osp.join(out_patient_dir,  f'{self.name}_Diastole_Labelmap.nii.gz'))
        io.write_metatensor_to_nifti(self.seg_sys, osp.join(out_patient_dir, f'{self.name}_Systole_Labelmap.nii.gz'))

    def viz_data(self, save_dir, gif, dur, aspect_ratio=1.7):
        """
        Visualizes the data and saves it as 2d images and GIF.

        Args:
            save_dir (str): Directory to save the visualizations.
            gif (bool): Whether to create a GIF.
            dur (int): Duration of each frame in the GIF (in milliseconds).
            aspect_ratio (float, optional): Aspect ratio for visualization. Defaults to 1.7.
        """
        save_dir = dirs.create_subdir(save_dir, self.name)
        img = self.img.as_tensor().swapaxes(3, 1)

        for t in range(img.shape[0]):
            if t == self.tsys:
                seg = self.seg_sys.as_tensor().swapaxes(3, 1).squeeze(0)
            elif t == self.tdia:
                seg = self.seg_dia.as_tensor().swapaxes(3, 1).squeeze(0)
            else:
                seg = None

            plots.save_image_mask_overlay(img[t], seg, osp.join(save_dir, f'{self.name}_{t}.png'), 0.3, aspect_ratio)

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

    def __getitem__(self, idx: int) -> Patient:
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
        self.loader = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx) -> Patient:
        patient_dir = self.data_dirs[idx]
        name = osp.basename(patient_dir)
        tdia, tsys = self.extract_times(patient_dir)

        # Load 4D image meta tensor [t,x,y,z]
        img = self.loader(osp.join(patient_dir, f'{name}_4d.nii.gz'))

        # Load segmentation masks [c,x,y,z]
        seg_dia = self.loader(osp.join(patient_dir,  f'{name}_frame{tdia:02d}_gt.nii.gz'))
        seg_sys = self.loader(osp.join(patient_dir, f'{name}_frame{tsys:02d}_gt.nii.gz'))

        return Patient(name, tsys, tdia, min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)

    def extract_times(self, data_dir):
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
