import torch
from natsort import natsorted
import glob
import os.path as osp
from tqdm import tqdm
from ml_collections import config_dict
import pandas as pd
from PIL import Image
from typing import Tuple
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from tabulate import tabulate

from svs.modules.datasets import FlowDataset, xyzt_to_tzyx, xyz_to_zyx, t3xyz_to_tzyx3, FlowPatient
from svs.utils import dirs, flow_utils, plots
from svs.utils.enums import FlowDirection
from opticalFlow_cuda_ext import opticalFlow
import svs.modules.transforms.functional as F


class Warp:
    """
    A class to perform 3D warping of volumetric data using optical flow.
    """

    def __init__(self, interpolation_type: str, boundary_type: str, mesh_length_type: str, lenghts: Tuple[int, int, int]):
        self.interpolation, self.boundary = flow_utils.get_interpolation_type(interpolation_type, boundary_type)
        self.mesh_length_type = mesh_length_type
        self.lz, self.ly, self.lx = lenghts

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Warps the input volume using the given flow field.

        Args:
            x (torch.Tensor): The 5D tensor representing the volume to be warped, with shape (batch_size, channels, depth, height, width).
            flow (torch.Tensor): The 5D tensor representing the flow field, with shape (batch_size, depth, height, width, 3). Consistent with pytorch grid_sample
            depth (int): The depth of the volume.
            height (int): The height of the volume.
            width (int): The width of the volume.

        Returns:
            torch.Tensor: The warped volume.
        """
        nz, ny, nx = x.shape[2:]
        lz, ly, lx = flow_utils.get_mesh_length(self.mesh_length_type, nz, ny, nx, self.lz, self.ly, self.lx)
        mesh_info = opticalFlow.MeshInfo3D(nz, ny, nx, lz, ly, lx)
        warp_op = opticalFlow.WarpingCNN3D(mesh_info, self.interpolation, self.boundary)
        x_w = warp_op.forward(x.contiguous(), u.contiguous())
        return x_w


class OpticalFlowWarper:
    def __init__(self, config: config_dict.ConfigDict):
        """
        Initializes the OpticalFlowWarper with the provided configuration.

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for optical flow processing.
        """
        self.cfg = config
        self.save_dir = dirs.create_timestamped_dir(self.cfg.data.out_dir, 'warp')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._check_device()
        self.dataset = self._initialize_dataset()
        self.warp = self._initialize_warp()

    def _check_device(self):
        """
        Checks if the CUDA device is available. Raises an error if not.
        """
        print(f'Warping\nSave dir: {self.save_dir}\nDevice: {self.device}')
        if self.device.type != 'cuda':
            raise RuntimeError("Optical flow computation requires a CUDA device.")

    def _initialize_dataset(self) -> FlowDataset:
        return FlowDataset(self.cfg.data.base_dir, self.cfg.data.imgs_dir, self.cfg.data.segs_dir, self.cfg.data.metadata_file,
                           self.cfg.data.flow_subdir.forward, self.cfg.data.flow_subdir.backward)

    def _initialize_warp(self) -> Warp:
        return Warp(self.cfg.warping.interpolation.type, self.cfg.warping.interpolation.boundary,
                    self.cfg.warping.mesh.type, self.cfg.warping.mesh.length)

    def process(self):
        """
        Processes all patients in the dataset.
        """
        pbar = tqdm(total=len(self.dataset))
        report = pd.DataFrame(columns=['Patient', 'Dice_bwd', 'Dice_fwd', 'HD_bwd', 'HD_fwd'])

        for i in range(len(self.dataset)):
            metrics = self._process_patient(self.dataset[i], pbar)
            report.loc[len(report)] = [self.dataset[i].name, metrics[0], metrics[1], metrics[2], metrics[3]]

        report.loc[len(report)] = ['Mean', report['Dice_bwd'].mean(), report['Dice_fwd'].mean(), report['HD_bwd'].mean(), report['HD_fwd'].mean()]

        pbar.close()
        self._save_config()

        # Print and save report
        report = report.round(3)
        print(tabulate(report, headers='keys', tablefmt='psql'))
        report.to_excel(osp.join(self.save_dir, 'metrics.xlsx'), index=False)

    def _process_patient(self, patient: FlowPatient, pbar: tqdm) -> Tuple[float, float, float, float]:
        """
        Warping masks using the optical flow for a single patient.

        Args:
            patient (FlowPatient): The patient in the dataset.
            pbar (tqdm): The progress bar object.
        """
        # Prepare data
        seg_dia = F.to_float_tensor(F.add_leading_dims(2, F.reorder_axes(xyz_to_zyx, patient.seg_dia_array()))).to(self.device)
        seg_sys = F.to_float_tensor(F.add_leading_dims(2, F.reorder_axes(xyz_to_zyx, patient.seg_sys_array()))).to(self.device)
        forward_flow = F.to_float_tensor(F.reorder_axes(t3xyz_to_tzyx3, patient.forward_flow)).to(self.device)
        backward_flow = F.to_float_tensor(F.reorder_axes(t3xyz_to_tzyx3, patient.backward_flow)).to(self.device).to(self.device)

        _, (initial_mask, final_mask), _ = flow_utils.compute_timepoints(patient.tdia, patient.tsys, seg_dia, seg_sys, FlowDirection.FORWARD)

        # Propagate masks
        forward_masks, backward_masks = self._propagate(initial_mask, final_mask, forward_flow, backward_flow)
        metrics = self._compute_metrics(forward_masks, backward_masks)

        # Visualize results
        if self.cfg.debug.viz:
            self._write_images(patient, forward_masks, backward_masks)

        pbar.update(1)
        return metrics

    def _propagate(self, initial_mask: torch.Tensor, final_mask: torch.Tensor, forward_flow: torch.Tensor,
                   backward_flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates masks using forward and backward optical flow.

        Args:
            mi (torch.Tensor): Initial mask.
            mf (torch.Tensor): Final mask.
            ff (torch.Tensor): Forward flow field.
            bf (torch.Tensor): Backward flow field.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Warped masks for forward and backward flows.
        """
        assert forward_flow.shape[0] == backward_flow.shape[0], "Forward and backward flow have different timesteps"
        forward_masks = [initial_mask]
        backward_masks = [final_mask]
        num_timesteps = forward_flow.shape[0]

        for t in range(num_timesteps):
            forward_masks.append(self.warp(forward_masks[-1], F.add_dim_at(0, forward_flow[t])))
            backward_masks.append(self.warp(backward_masks[-1], F.add_dim_at(0, backward_flow[t])))

        backward_masks.reverse()  # mi_est -> mf

        forward_masks = torch.where(torch.cat(forward_masks, dim=0) > 0.5, 1.0, 0.0)
        backward_masks = torch.where(torch.cat(backward_masks, dim=0) > 0.5, 1.0, 0.0)
        return forward_masks, backward_masks

    def _compute_metrics(self, forward_masks: torch.Tensor, backward_masks: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Computes the Dice and Hausdorff distance metrics for forward and backward propagated masks.

        Args:
            forward_masks (torch.Tensor): Tensor containing the masks propagated forward in time.
            backward_masks (torch.Tensor): Tensor containing the masks propagated backward in time.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the Dice and Hausdorff distance metrics for both forward and backward propagation.
        """
        # Initital masks, groundtruth and prediction
        initial_gt = F.add_dim_at(0, forward_masks[0])
        initial_pred = F.add_dim_at(0, backward_masks[0])
        # Final masks, groundtruth and prediction
        final_gt = F.add_dim_at(0, backward_masks[-1])
        final_pred = F.add_dim_at(0, forward_masks[-1])

        # Metrics between the initial mask and estimation (final mask propagation)
        dice_bwd = compute_dice(initial_pred, initial_gt, include_background=False).mean().item()
        hausdorff_bwd = compute_hausdorff_distance(initial_pred, initial_gt, include_background=False).mean().item()
        # Metrics between the final mask and estimation (initial mask propagation)
        dice_fwd = compute_dice(final_pred, final_gt, include_background=False).mean().item()
        hausdorff_fwd = compute_hausdorff_distance(final_pred, final_gt, include_background=False).mean().item()

        return dice_bwd, dice_fwd, hausdorff_bwd, hausdorff_fwd

    def _write_images(self, patient: FlowPatient, forward_masks: torch.Tensor, backward_masks: torch.Tensor):
        """
        Writes a series of images overlaid with forward and backward masks for visualization.

        Args:
            patient (FlowPatient): The patient data.
            forward_masks (torch.Tensor): Tensor containing the forward masks.
            backward_masks (torch.Tensor): Tensor containing the backward masks.

        """
        patient_dir = dirs.create_subdir(self.save_dir, osp.join('images', patient.name))

        *_, times_fwd = flow_utils.compute_timepoints(patient.tdia, patient.tsys, None, None, FlowDirection.FORWARD)
        img4d = F.reorder_axes(xyzt_to_tzyx, patient.img_array())
        forward_masks = forward_masks.cpu().numpy().squeeze()
        backward_masks = backward_masks.cpu().numpy().squeeze()

        fcolor = self.cfg.debug.forward_color
        bcolor = self.cfg.debug.backward_color

        for t in range(forward_masks.shape[0]):
            img3d = img4d[times_fwd[t]]

            if t == 0:
                # Groundtruth and estimated mask
                masks = [forward_masks[t], F.erode(forward_masks[t]), F.erode(backward_masks[t])]
                alphas = [0.2, 1.0, 1.0]
                colors = [fcolor, fcolor, bcolor]
            elif t == forward_masks.shape[0]-1:
                masks = [backward_masks[t], F.erode(backward_masks[t]), F.erode(forward_masks[t])]
                alphas = [0.2, 1.0, 1.0]
                colors = [bcolor, bcolor, fcolor]
            else:
                masks = [F.erode(forward_masks[t]), F.erode(backward_masks[t])]
                alphas = [1.0, 1.0]
                colors = [fcolor, bcolor]

            plots.write_image_mask_overlay(img3d, masks, osp.join(
                patient_dir, f'im_t{times_fwd[t]}.png'), alphas, colors, self.cfg.debug.aspect_ratio)

        if self.cfg.debug.gif:
            frames = [Image.open(image) for image in natsorted(glob.glob(f'{patient_dir}/*.png'))]
            frame_one = frames[0]
            frame_one.save(osp.join(patient_dir, 'animation.gif'), format='GIF', append_images=frames,
                           save_all=True, duration=self.cfg.debug.dur, loop=0)

    def _save_config(self):
        """
        Saves the configuration used for this run to a JSON file.
        """
        with open(osp.join(self.save_dir, "config.json"), "w") as f:
            f.write(self.cfg.to_json(indent=4))
