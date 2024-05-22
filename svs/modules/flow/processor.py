import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from ml_collections import config_dict
from typing import List

from svs.modules.datasets import MRIDataset, xyzt_to_tzyx
from svs.utils import dirs, flow_utils
from svs.modules.flow.tvl13d import TVL13DOpticalFlow


class OpticalFlowProcessor:
    def __init__(self, config: config_dict.ConfigDict):
        """
        Initializes the OpticalFlowProcessor with the provided configuration.

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for optical flow processing.
        """
        self.cfg = config
        self.save_dir = dirs.create_timestamped_dir(self.cfg.data.out_dir, f'flow-{self.cfg.flow.mode}')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._check_device()
        self.dataset = self._initialize_dataset()
        self.optflow = self._initialize_optflow()

    def _check_device(self):
        """
        Checks if the CUDA device is available. Raises an error if not.
        """
        print(f'TVL1-3D Optical flow\nSave dir: {self.save_dir}\nMode: {self.cfg.flow.mode}\nDevice: {self.device}')
        if self.device.type != 'cuda':
            raise RuntimeError("Optical flow computation requires a CUDA device.")

    def _initialize_dataset(self) -> MRIDataset:
        """
        Initializes the MRI dataset.

        Returns:
            MRIBaseDataset: The initialized MRI dataset.
        """
        return MRIDataset(self.cfg.data.base_dir, self.cfg.data.imgs_dir, self.cfg.data.segs_dir, self.cfg.data.metadata_file)

    def _initialize_optflow(self) -> TVL13DOpticalFlow:
        """
        Initializes the TVL1-3D Optical Flow object.

        Returns:
            TVL13DOpticalFlow: The initialized TVL1-3D Optical Flow object.
        """
        return TVL13DOpticalFlow(
            self.cfg.flow.num_scales,
            self.cfg.flow.max_warps,
            self.cfg.flow.max_outer_iterations,
            self.cfg.flow.weight_matching,
            self.cfg.flow.weight_tv,
            self.cfg.flow.primal_dual.type,
            self.cfg.flow.primal_dual.params,
            self.cfg.flow.anisotropic_diff.enabled,
            self.cfg.flow.anisotropic_diff.params,
            self.cfg.flow.gaussian_blur.enabled,
            self.cfg.flow.gaussian_blur.sigma,
            self.cfg.flow.interpolation.type,
            self.cfg.flow.interpolation.boundary,
            self.cfg.flow.mesh.type,
            self.cfg.flow.mesh.length,
            self.device
        )

    def process(self):
        """
        Processes the optical flow for all patients in the dataset.
        """
        indices = self._get_indices()
        print(f'Computing flow from patients: {indices[0]} to {indices[-1]}')
        pbar = tqdm(total=len(indices))

        for i in indices:
            self._process_patient(i, pbar)

        pbar.close()
        self._save_config()

    def _get_indices(self) -> List[int]:
        """
        Determines the indices of the dataset to process.

        Returns:
            List[int]: The list of indices to process.
        """
        idxs = eval(self.cfg.flow.indices)
        return np.arange(idxs[0], idxs[1], 1).tolist() if -1 not in idxs else np.arange(len(self.dataset)).tolist()

    def _process_patient(self, i: int, pbar: tqdm):
        """
        Processes the optical flow for a single patient.

        Args:
            i (int): The index of the patient in the dataset.
            pbar (tqdm): The progress bar object.
        """
        img = torch.from_numpy(self.dataset[i].img_array().transpose(xyzt_to_tzyx)).float().to(self.device)
        *_, times = flow_utils.compute_timepoints(self.dataset[i].tdia, self.dataset[i].tsys, None, None, self.cfg.flow.mode)
        nt, nz, ny, nx = img.shape
        u = torch.zeros((nz, ny, nx, 3), dtype=torch.float32, device=self.device)
        us = torch.zeros((len(times) - 1, 3, nx, ny, nz), dtype=torch.float32, device=self.device)
        p = torch.zeros((nz, ny, nx, 3, 3), dtype=torch.float32, device=self.device)

        for j in range(len(times) - 1):
            I0 = img[times[j + 1]]
            I1 = img[times[j]]
            pbar.set_postfix_str(f'P: {self.dataset[i].name}, ({times[j]}->{times[j + 1]}/{times[-1]})')
            u, p = self.optflow.compute(I0, I1, u, p)

            if self.cfg.flow.median_filter.enabled:
                u = flow_utils.apply_median_filter(u, self.cfg.flow.median_filter.kernel, self.device)
            us[j] = u.permute(3, 2, 1, 0)  # change to (3, x, y, z)

        np.save(osp.join(self.save_dir, f'{self.dataset[i].name}_{self.cfg.flow.mode}_flow.npy'), us.cpu().detach().numpy())
        pbar.update(1)

    def _save_config(self):
        """
        Saves the configuration used for this run to a JSON file.
        """
        with open(osp.join(self.save_dir, "config.json"), "w") as f:
            f.write(self.cfg.to_json(indent=4))
