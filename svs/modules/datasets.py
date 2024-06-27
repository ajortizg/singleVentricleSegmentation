import os
import numpy as np
import nibabel as nib
import os.path as osp
import pandas as pd
from typing import Optional, Union, Any, List, Dict
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from PIL import Image
import logging
import glob
from natsort import natsorted
from collections import defaultdict
import torch.nn.functional as F
import torch
import lightning as pl

from svs.utils import dirs, plots
from svs.utils.constants import *
from svs.utils.enums import FlowDirection, NNDatasetMode
from svs.utils.flow_utils import compute_timepoints


xyzt_to_zyxt = (2, 1, 0, 3)
zyxt_to_xyzt = (2, 1, 0, 3)

xyz_to_zyx = (2, 1, 0)
zyx_to_xyz = (2, 1, 0)

xyzt_to_tzyx = (3, 2, 1, 0)
tzyx_to_xyzt = (3, 2, 1, 0)

# Optical flow axes reordering
t3xyz_to_t3zyx = (0, 1, 4, 3, 2)
t3xyz_to_tzyx3 = (0, 4, 3, 2, 1)
t3zyx_to_t3xyz = (0, 1, 4, 3, 2)


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
        Returns the 4D image data as a NumPy array with shape [x,y,z,t]
        """
        return self.img.get_fdata() if self.img else np.array([])

    def seg_dia_array(self) -> np.ndarray:
        """
        Returns the end-diastole segmentation mask data as a NumPy array with shape [x,y,z]
        """
        return self.seg_dia.get_fdata() if self.seg_dia else np.array([])

    def seg_sys_array(self) -> np.ndarray:
        """
        Returns the end-systole segmentation mask data as a NumPy array with shape [x,y,z]
        """
        return self.seg_sys.get_fdata() if self.seg_sys else np.array([])

    def img_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
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

    def seg_dia_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
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

    def seg_sys_from_array(self, x: np.ndarray, affine: nib.Nifti1Image, header: nib.Nifti1Header, update_shape: bool, zooms=None):
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

    def write_nifti(self, out_img_dir, out_seg_dir):
        """
        Saves the Nifti images for the 4D image and segmentation masks to disk.

        Args:
            out_img_dir (str): Directory to save the 4D image.
            out_seg_dir (str): Directory to save the segmentation masks.
        """
        out_patient_dir = dirs.create_subdir(out_seg_dir, self.name)
        nib.save(self.img, osp.join(out_img_dir, f"{self.name}.nii.gz"))
        nib.save(self.seg_dia, osp.join(out_patient_dir,  f"{self.name}_Diastole_Labelmap.nii.gz"))
        nib.save(self.seg_sys, osp.join(out_patient_dir, f"{self.name}_Systole_Labelmap.nii.gz"))

    def viz_data(self, save_dir, gif, dur, alpha=0.3, color=[1, 1, 0], aspect_ratio=1.7):
        """
        Visualizes the data and saves it as images or a GIF.

        Args:
            save_dir (str): Directory to save the visualizations.
            gif (bool): Whether to create a GIF.
            dur (int): Duration of each frame in the GIF (in milliseconds).
            aspect_ratio (float, optional): Aspect ratio for visualization.
        """
        save_dir = dirs.create_subdir(save_dir, self.name)
        img = self.img_array().transpose(xyzt_to_tzyx)
        for t in range(img.shape[0]):
            if t == self.tsys:
                seg = self.seg_sys_array().swapaxes(2, 0)
            elif t == self.tdia:
                seg = self.seg_dia_array().swapaxes(2, 0)
            else:
                seg = None

            plots.write_image_mask_overlay(img[t], seg, osp.join(save_dir, f"{self.name}_{t}.png"), alpha, color, aspect_ratio)

        if gif:
            frames = [Image.open(image) for image in natsorted(glob.glob(f"{save_dir}/*.png"))]
            frame_one = frames[0]
            frame_one.save(osp.join(save_dir, "animation.gif"), format="GIF", append_images=frames,
                           save_all=True, duration=dur, loop=0)


class MRIDataset(Dataset):
    def __init__(self, base_dir: str, imgs_dir: str, segs_dir: str, metadata_file: str):
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
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Patient:
        row = self.df.iloc[idx]
        name = row["Name"]
        tsys = row["Systole"]
        tdia = row["Diastole"]

        # Load 4D image [x,y,z,t]
        img = nib.load(osp.join(self.imgs_dir, f"{name}.nii.gz"))

        # Load segmentation masks [x,y,z]
        seg_dia = nib.load(osp.join(self.segs_dir, name, f"{name}_Diastole_Labelmap.nii.gz"))
        seg_sys = nib.load(osp.join(self.segs_dir, name, f"{name}_Systole_Labelmap.nii.gz"))

        return Patient(name, tsys, tdia,  min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)


class RawACDCDadataset(Dataset):
    def __init__(self, base_dir: str):
        """
        Initializes the dataset by listing all patient directories.

        Args:
            base_dir (str): The base directory containing patient data.
        """
        self.data_dirs = sorted([osp.join(base_dir, d) for d in os.listdir(base_dir) if osp.isdir(osp.join(base_dir, d))])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx) -> Patient:
        patient_dir = self.data_dirs[idx]
        name = osp.basename(patient_dir)
        tdia, tsys = self.extract_diastole_systole_times(patient_dir)

        # Load 4D image [x,y,z,t]
        img = nib.load(osp.join(patient_dir, f"{name}_4d.nii.gz"))

        # Load segmentation masks [x,y,z]
        seg_dia = nib.load(osp.join(patient_dir,  f"{name}_frame{tdia:02d}_gt.nii.gz"))
        seg_sys = nib.load(osp.join(patient_dir, f"{name}_frame{tsys:02d}_gt.nii.gz"))

        return Patient(name, tsys, tdia,  min(tdia, tsys), max(tdia, tsys), img, seg_dia, seg_sys)

    def extract_diastole_systole_times(self, data_dir):
        """
        Extracts the diastole and systole times from the Info.cfg file.

        Args:
            data_dir (str): The directory containing the Info.cfg file.

        Returns:
            Tuple[int, int]: The diastole and systole times.
        """
        with open(osp.join(data_dir, "Info.cfg"), "r") as f:
            tdia = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
            tsys = int(f.readline().replace(' ', '').replace('\n', '').split(':')[1])
        return tdia, tsys


@dataclass
class FlowPatient(Patient):
    """
    Data class representing a patient with additional flow data.
    """
    forward_flow: np.ndarray = field(default_factory=lambda: np.array([]))  # (t, 3, x, y, z)
    backward_flow: np.ndarray = field(default_factory=lambda: np.array([]))


class FlowDataset(MRIDataset):
    """
    Dataset with additional optical flow data
    """

    def __init__(
        self,
        base_dir: str,
        imgs_dir: str,
        segs_dir: str,
        metadata_file: str,
        forward_flow_subdir: str = 'forward',
        backward_flow_subdir: str = 'backward'
    ):
        super().__init__(base_dir, imgs_dir, segs_dir, metadata_file)
        self.forward_flow_subdir = forward_flow_subdir
        self.backward_flow_subdir = backward_flow_subdir

    def __getitem__(self, idx: int) -> FlowPatient:
        patient = super().__getitem__(idx)

        # Read optical flow
        forward_flow = self._load_flow(patient.name, FlowDirection.FORWARD, self.forward_flow_subdir)
        backward_flow = self._load_flow(patient.name, FlowDirection.BACKWARD, self.backward_flow_subdir)

        return FlowPatient(name=patient.name, tsys=patient.tsys, tdia=patient.tdia,
                           init_ts=patient.init_ts, final_ts=patient.final_ts,
                           img=patient.img, seg_dia=patient.seg_dia, seg_sys=patient.seg_sys,
                           forward_flow=forward_flow, backward_flow=backward_flow)

    def _load_flow(self, patient_name: str, direction: Union[str, FlowDirection], flow_subdir: str) -> np.ndarray:
        """
        Loads the optical flow data for the specified patient and direction.

        Args:
            patient_name (str): Name of the patient.
            direction (Union[str, FlowDirection]): Direction of the flow ('forward' or 'backward').
            flow_subdir (str): Sub-directory containing the flow data.

        Returns:
            np.ndarray: The optical flow data as a NumPy array with shape (t, 3, x, y, z).
        """
        return np.load(osp.join(self.base_dir, flow_subdir, f"{patient_name}_{direction}_flow.npy"))


class NNDataset(FlowDataset):
    """
    Dataset used for Neural Network (NN) training.
    """

    def __init__(
        self,
        base_dir: str,
        imgs_dir: str,
        segs_dir: str,
        metadata_file: str,
        split: NNDatasetMode | str,
        forward_flow_subdir: str,
        backward_flow_subdir: str,
        transforms: Any = None,
    ):
        super().__init__(base_dir, imgs_dir, segs_dir, metadata_file, forward_flow_subdir, backward_flow_subdir)

        if split != NNDatasetMode.COMPLETE:
            self.df = self.df[self.df["Split"] == split]
            self.df.reset_index(inplace=True, drop=True)

        self.transforms = transforms

    def __getitem__(self, idx: int):
        patient = super().__getitem__(idx)

        (ti, tf), (mi, mf), _ = compute_timepoints(
            patient.tdia,
            patient.tsys,
            patient.seg_dia_array(),
            patient.seg_sys_array(),
            FlowDirection.FORWARD
        )

        # Group patient data into a dict
        # Array shape format is [X, Y, Z, [T]]. For optical flow [T, 3, X, Y, Z]
        data = {
            PATIENT_NAME_KEY: patient.name,
            IMAGE_KEY: torch.from_numpy(patient.img_array()),
            MI_KEY: torch.from_numpy(mi),
            TI_KEY: torch.tensor(ti, dtype=torch.int),
            MF_KEY: torch.from_numpy(mf),
            TF_KEY: torch.tensor(tf, dtype=torch.int),
            FWD_FLOW_KEY: torch.from_numpy(patient.forward_flow),
            BWD_FLOW_KEY: torch.from_numpy(patient.backward_flow),
            TED_KEY: torch.tensor(patient.tdia, dtype=torch.int),
            TES_KEY: torch.tensor(patient.tsys, dtype=torch.int),
            MED_KEY: torch.tensor(patient.seg_dia_array()),
            MES_KEY: torch.tensor(patient.seg_sys_array())
        }

        # Apply transformations to data
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle variable-length sequences in the batch.

        This function pads tensors to the maximum length in the batch for optical flow,
        and calculates time sequences based on initial and final timestamps.

        Parameters:
            batch (List[Dict]): List of samples from the dataset.

        Returns:
            Dict: Collated batch with padded sequences.
        """
        # Find the maximum time dimension "t" in the batch
        max_t = max(sample[FWD_FLOW_KEY].shape[0] for sample in batch)

        collated_batch = defaultdict(list)

        for sample in batch:
            pad_t = None
            for key, value in sample.items():
                if key in [FWD_FLOW_KEY, BWD_FLOW_KEY]:
                    pad_t = max_t - value.shape[0]
                    collated_batch[key].append(F.pad(value, (0, 0,
                                                             0, 0,
                                                             0, 0,
                                                             0, 0,
                                                             0, pad_t)))
                else:
                    collated_batch[key].append(value)

            if pad_t is not None:
                collated_batch[OFFSET_KEY].append(pad_t)

        # Keep non-tensor values as lists
        for key in collated_batch.keys():
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key], dim=0)

        # Create time sequences
        ti = collated_batch[TI_KEY]
        tf = collated_batch[TF_KEY]
        max_time_gap = (tf - ti).max().item()
        times_fwd = [ti]
        times_bwd = [tf]

        for _ in range(max_time_gap):
            next_time = times_fwd[-1] + 1
            times_fwd.append(torch.where(next_time > tf, tf, next_time))

            prev_time = times_bwd[-1] - 1
            times_bwd.append(torch.where(prev_time < ti, ti, prev_time))

        collated_batch[FWD_TS_KEY] = torch.stack(times_fwd, dim=1)
        collated_batch[BWD_TS_KEY] = torch.stack(times_bwd, dim=1)

        return collated_batch


class LitNNDataset(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        train_config: Dict[str, Any],
        val_config: Dict[str, Any]
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_config = train_config
        self.val_config = val_config

    def setup(self, stage: str):
        if stage == "fit":
            self.ds_trn = NNDataset(**self.train_config)
            self.ds_val = NNDataset(**self.val_config)
        elif stage in {"test", "predict"}:
            raise NotImplementedError(f"LitNNDataset.setup {stage} functionality not implemented yet.")

    def train_dataloader(self):
        return DataLoader(
            self.ds_trn,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=NNDataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=NNDataset.collate_fn
        )

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, idx: int) -> Dict[str, Any]:
        for k in [IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY]:
            assert isinstance(batch[k], torch.Tensor), f"transfer_batch_to_device failed because {k} is not a torch Tensor."
            batch[k] = batch[k].to(device)
        return batch
