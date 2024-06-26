import torch
from typing import Any, Dict, Iterable, Tuple
import numpy as np
import torch.nn.functional as F

from svs.modules.transforms.base import BaseTransform
from svs.modules.transforms.functional import rotx, roty, rotz
from svs.utils.constants import *


class RandomRotate(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        spatial_size: Iterable[int],
        rot_ranges: Iterable[Iterable[float]],
        boundaries: Dict[str, str],
        modes: Dict[str, str],
        **kwargs
    ):
        """
        Initialize the RandomRotate transformation class for randomly rotating
        3D volumes based on the specified parameters.

        Args:
            keys (Iterable[str]): Keys to apply the transformation.
            p (float): Probability of applying the rotation.
            spatial_size (Iterable[int]): Spatial size of the volume (Z, Y, X).
            rot_ranges (Iterable[Iterable[float]]): Rotation ranges for each axis (X, Y, Z) in degrees.
            boundaries (Iterable[str]): Boundary modes for grid_sample (e.g., 'zeros', 'border', 'reflection').
            modes (Iterable[str]): Interpolation modes for grid_sample (e.g., 'bilinear', 'nearest').
            kwargs: Additional keyword arguments for torch.nn.functional.grid_sample
        """
        super().__init__(keys)

        assert len(spatial_size) == 3, "Spatial size must have exactly 3 elements."
        assert len(rot_ranges) == 3, "Rotation ranges must have exactly 3 elements."

        self.p = p
        self.spatial_size = spatial_size
        self.range_x = rot_ranges[0]
        self.range_y = rot_ranges[1]
        self.range_z = rot_ranges[2]
        self.boundaries = boundaries
        self.modes = modes
        self.kwargs = kwargs

        self.identity_grid = self.get_identity_grid()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() < self.p:
            # The same rotation is appled to all data
            self.R = self.sample_rotation_matrix()
            self.rotated_grid = torch.matmul(self.identity_grid, self.R).float()

            return super().apply_transform(data)

    def _transform_impl(self, x: torch.Tensor, key: str, metadata: Dict[str, Any] | None = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The data element to transform.
            key (str): The key associated with the data element.
            metadata (Dict[str, Any] | None): Optional metadata providing additional context for transformation.

            Expecting the following shapes:
                - images        (T, C, Z, Y, X)
                - masks         (C, Z, Y, X)
                - optical flow  (T, 3, Z, Y, X)
        """
        mode = self.modes[key]
        boundary = self.boundaries[key]
        add_batch_dim = x.ndim == 4 and key in MASKS_KEYS

        if add_batch_dim:
            x = torch.unsqueeze(x, dim=0)  # Add dummy batch dim for masks

        x_r = F.grid_sample(
            x,
            self.rotated_grid.repeat((x.shape[0], *[1]*self.rotated_grid.ndim)),
            mode=mode,
            padding_mode=boundary,
            **self.kwargs
        )

        if key in FLOWS_KEYS:
            x_r = self.flow_transform(x_r)

        if add_batch_dim:
            x_r = torch.squeeze(x_r, dim=0)

        return x_r

    def get_identity_grid(self) -> torch.Tensor:
        """
        Create a normalized identity coordinates grid for the given spatial size.

        Returns:
            torch.Tensor: Identity grid with shape (Z, Y, X, 3).
        """
        space = [torch.linspace(-1, 1, s) for s in self.spatial_size[::-1]]
        grid = torch.meshgrid(space, indexing="ij")
        grid = torch.stack(grid, -1)
        spatial_dims = list(range(len(self.spatial_size)))
        grid = grid.permute((*spatial_dims[::-1], len(self.spatial_size)))
        return grid

    def sample_rotation_matrix(self) -> torch.Tensor:
        Rx = rotx(np.random.uniform(self.range_x[0], self.range_x[1]))
        Ry = roty(np.random.uniform(self.range_y[0], self.range_y[1]))
        Rz = rotz(np.random.uniform(self.range_z[0], self.range_z[1]))
        R = Rz @ Ry @ Rx
        return torch.from_numpy(R).float()

    def flow_transform(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Transforms the 3D optical flow according to the rotation matrix. This rotation
        adjusts the flow vectors to be consistent with the transformed volume.

        Args:
            flow (torch.Tensor): Optical flow tensor with shape (T, 3, Z, Y, X),
                                where 3 represents the flow components along X, Y, Z axes.

        Returns:
            torch.Tensor: Transformed flow tensor with the same shape.
        """
        flow_r = torch.einsum('ij,tjxyz->tixyz', self.R, flow)
        return flow_r
