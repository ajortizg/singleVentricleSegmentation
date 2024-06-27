import numpy as np
import torch
from typing import Iterable, Dict, Any, Tuple

from svs.modules.transforms.base import BaseTransform


class AdditiveGaussianNoise(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        mu: float,
        sigma_range: Iterable[float]
    ):
        super().__init__(keys)

        assert len(sigma_range) == 2, "sigma_range must have 2 elements."

        self.p = p
        self.mu = mu
        self.sigma_range = sigma_range

    def __call__(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(
        self,
        x: torch.Tensor,
        key: str,
        metadata: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = torch.randn_like(x) * sigma + self.mu
        return x + noise


class GammaCorrection(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        gamma_range: Iterable[float],
        invert_image: bool,
        retain_stats: bool
    ):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors).

        Args:
            keys (Iterable[str]): Keys to apply the transformation.
            p (float): Probability of applying gamma correction.
            gamma_range (Tuple[float, float]): Range to sample gamma from.
            invert_image (bool): Whether to invert the image before applying gamma augmentation.
            retain_stats (bool): Whether to retain the original mean and standard deviation of the image.
        """
        super().__init__(keys)

        assert len(gamma_range) == 2, "sigma_range must have 2 elements."

        self.p = p
        self.retain_stats = retain_stats
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.epsilon = 1e-7

    def __call__(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(
        self,
        x: torch.Tensor,
        key: str,
        metadata: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        if self.invert_image:
            x = -x
        if self.retain_stats:
            mean = x.mean()
            std = x.std()

        if np.random.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = np.random.uniform(self.gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])

        minm = x.min()
        rnge = x.max() - minm
        x = torch.pow(((x - minm) / (rnge + self.epsilon)), gamma) * (rnge + self.epsilon) + minm

        if self.retain_stats:
            x = (x - x.mean()) / (x.std() + self.epsilon) * std + mean
        if self.invert_image:
            x = -x
        return x


class ContrastAugmentation(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        contrast_range: Tuple[float, float],
        preserve_range: bool
    ):
        """
        Args:
            keys (Iterable[str]): Keys to apply the transformation.
            p (float): Probability of applying contrast augmentation.
            contrast_range (Tuple[float, float]): Range to sample the contrast factor from.
            preserve_range (bool): Whether to preserve the original value range after contrast adjustment.
        """
        super().__init__(keys)
        self.p = p
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range

    def __call__(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(
        self,
        x: torch.Tensor,
        key: str,
        metadata: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        if np.random.random() < 0.5 and self.contrast_range[0] < 1:
            factor = np.random.uniform(self.contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])

        mn = x.mean()
        if self.preserve_range:
            minm = x.min()
            maxm = x.max()

        x = (x - mn) * factor + mn

        if self.preserve_range:
            x[x < minm] = minm
            x[x > maxm] = maxm
        return x


class MultiplicativeScaling(BaseTransform):
    def __init__(
        self,
        keys: Iterable[str],
        p: float,
        scale_range: Tuple[float, float]
    ):
        super().__init__(keys)
        self.p = p
        self.scale_range = scale_range

    def __call__(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if np.random.uniform() < self.p:
            data = super().apply_transform(data)
        return data

    def _transform_impl(
        self,
        x: torch.Tensor,
        key: str,
        metadata: Dict[str, Any] | None = None
    ) -> torch.Tensor:
        sigma = np.random.uniform(self.scale_range[0], self.scale_range[1])
        x *= sigma
        return x
