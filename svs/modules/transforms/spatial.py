import elasticdeform as ed
import torch
from typing import Any, Dict, Iterable, Tuple, List
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
        deform_shape: Iterable[int],
        rot_ranges: Iterable[Iterable[float]],
        boundary: str,
        modes: Dict[str, str],
        **kwargs
    ):
        """
        Initialize the RandomRotate transformation class for randomly rotating
        3D volumes based on the specified parameters.

        Args:
            keys (Iterable[str]): Keys to apply the transformation.
            p (float): Probability of applying the rotation.
            deform_shape (Iterable[int]): Spatial size of the volume (Z, Y, X).
            rot_ranges (Iterable[Iterable[float]]): Rotation ranges for each deform_axis (X, Y, Z) in degrees.
            boundary (str): Boundary modes for grid_sample (e.g., 'zeros', 'border', 'reflection').
            modes (Iterable[str]): Interpolation modes for grid_sample (e.g., 'bilinear', 'nearest').
            kwargs: Additional keyword arguments for torch.nn.functional.grid_sample
        """
        super().__init__(keys)

        assert len(deform_shape) == 3, "Spatial size must have exactly 3 elements."
        assert len(rot_ranges) == 3, "Rotation ranges must have exactly 3 elements."

        self.p = p
        self.deform_shape = deform_shape
        self.range_x = rot_ranges[0]
        self.range_y = rot_ranges[1]
        self.range_z = rot_ranges[2]
        self.boundary = boundary
        self.modes = modes
        self.kwargs = kwargs

        self.identity_grid = self.get_identity_grid()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            # The same rotation is appled to all data
            self.R = self.sample_rotation_matrix()
            self.rotated_grid = torch.matmul(self.identity_grid, self.R).float()
            data = super().apply_transform(data)
        return data

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
        add_batch_dim = x.ndim == 4 and key in MASKS_KEYS

        if add_batch_dim:
            x = torch.unsqueeze(x, dim=0)  # Add dummy batch dim for masks

        x_r = F.grid_sample(
            x,
            self.rotated_grid.repeat((x.shape[0], *[1]*self.rotated_grid.ndim)),
            mode=self.modes[key],
            padding_mode=self.boundary,
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
        space = [torch.linspace(-1, 1, s) for s in self.deform_shape[::-1]]
        grid = torch.meshgrid(space, indexing="ij")
        grid = torch.stack(grid, -1)
        spatial_dims = list(range(len(self.deform_shape)))
        grid = grid.permute((*spatial_dims[::-1], len(self.deform_shape)))
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


class RandomFlip(BaseTransform):
    def __init__(self, keys: Iterable[str], p: float, deform_axis: int):
        """
        Args:
            deform_axis (int): 2, 3, 4 for depth, vertical and horizontal flips
        """
        super().__init__(keys)

        assert deform_axis in {2, 3, 4}, "Axis must be 2, 3 or 4"

        self.p = p
        self.deform_axis = deform_axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x: torch.Tensor, key: str, metadata: Dict[str, Any] | None = None) -> torch.Tensor:
        add_batch_dim = x.ndim == 4 and key in MASKS_KEYS

        if add_batch_dim:
            x = torch.unsqueeze(x, dim=0)  # Add dummy batch dim for masks

        x_f = torch.flip(x, dims=(self.deform_axis,))

        if key in FLOWS_KEYS:
            x_f = self.flow_transform(x_f)
        if add_batch_dim:
            x_f = torch.squeeze(x_f, dim=0)

        return x_f

    def flow_transform(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow (torch.Tensor): Tensor with shape(T, 3, Z, Y, X)
        """
        index = {2: 2, 3: 1, 4: 0}[self.deform_axis]
        flow[:, index] *= -1
        return flow


class ElasticDeformation(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        deform_shape: Iterable[int],
        deform_axis: Iterable[int],
        sigma_range: Iterable[float],
        points: int,
        boundary: str,
        order: Dict[str, int]
    ):
        """
        Args:
            boundary: ({nearest, wrap, reflect, mirror, constant})
            order: {0, 1, 2, 3, 4}
        """
        super().__init__(keys)

        assert len(deform_shape) in {1, 2, 3}, "deform_shape must have 1, 2, or 3 elements."
        assert len(sigma_range) == 2, "sigma_range must have 2 elements."
        assert len(deform_axis) in {1, 2, 3}, "deform_axis must contain 1, 2, or 3 elements."
        assert len(deform_axis) == len(deform_shape), "deform_axis and deform_shape must have the same length."

        self.p = p
        self.deform_shape = deform_shape
        self.deform_axis = deform_axis
        self.sigma_range = sigma_range
        self.points = points
        self.boundary = boundary
        self.order = order

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            self.displacement = self.create_displacement()
            data = super().apply_transform(data)
        return data

    def _transform_impl(self, x: torch.Tensor, key: str, metadata: Dict[str, Any] | None = None):
        add_batch_dim = x.ndim == 4 and key in MASKS_KEYS

        if add_batch_dim:
            x = torch.unsqueeze(x, dim=0)  # Add dummy batch dim for masks

        x_d = ed.deform_grid(
            x.numpy(),
            self.displacement,
            order=self.order[key],
            mode=self.boundary,
            prefilter=False,
            axis=self.deform_axis
        )

        x_d = torch.from_numpy(x_d).float()
        if add_batch_dim:
            x_d = torch.squeeze(x_d, dim=0)
        return x_d

    def create_displacement(self) -> np.ndarray:
        points = [self.points] * len(self.deform_shape)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        return np.random.randn(len(self.deform_shape), *points) * sigma
