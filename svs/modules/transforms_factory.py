from typing import Dict, Any

import svs.modules.transforms as T
from svs.utils.constants import *
from svs.modules.datasets import (
    xyzt_to_tzyx,
    xyz_to_zyx,
    t3xyz_to_t3zyx,
    t3zyx_to_tzyx3
)


class TransformsFactory:
    def __init__(self, trn_config: Dict[str, Any]):
        self.trn_config = trn_config

    def get_train_transforms(self):
        return T.Compose([
            T.EnsureFloat(keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY]),

            # Reorder axes for tensors
            T.ReorderAxes(keys=[IMAGE_KEY], axes=xyzt_to_tzyx),
            T.ReorderAxes(keys=[MI_KEY, MF_KEY], axes=xyz_to_zyx),
            T.ReorderAxes(keys=[FWD_FLOW_KEY, BWD_FLOW_KEY], axes=t3xyz_to_t3zyx),

            # Add channel dimension
            T.AddDimAt(keys=[MI_KEY, MF_KEY], axis=0),
            T.AddDimAt(keys=[IMAGE_KEY], axis=1),

            # Spatial transformations
            T.RandomRotate(
                keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY],
                modes={IMAGE_KEY: "bilinear", MI_KEY: "nearest", MF_KEY: "nearest", FWD_FLOW_KEY: "bilinear", BWD_FLOW_KEY: "bilinear"},
                **self.trn_config["rotation"],
                align_corners=False
            ),
            T.RandomFlip(
                keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY],
                **self.trn_config["flip_d"]
            ),
            T.RandomFlip(
                keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY],
                **self.trn_config["flip_v"]
            ),
            T.RandomFlip(
                keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY],
                **self.trn_config["flip_h"]
            ),
            T.ElasticDeformation(
                keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY],
                **self.trn_config["elastic_deform"],
                order={IMAGE_KEY: 3, MI_KEY: 0, MF_KEY: 0, FWD_FLOW_KEY: 3, BWD_FLOW_KEY: 3}
            ),

            # Intensity transformations
            T.GammaCorrection(
                keys=[IMAGE_KEY],
                **self.trn_config["gamma_correction"]
            ),
            T.ContrastAugmentation(
                keys=[IMAGE_KEY],
                **self.trn_config["contrast"]
            ),
            T.MultiplicativeScaling(
                keys=[IMAGE_KEY],
                **self.trn_config["multiplicative_scaling"]
            ),
            T.AdditiveScaling(
                keys=[IMAGE_KEY],
                **self.trn_config["additive_scaling"]
            ),
            T.GaussianBlur(
                keys=[IMAGE_KEY],
                **self.trn_config["gaussian_blur"]
            ),
            T.AdditiveGaussianNoise(
                keys=[IMAGE_KEY],
                **self.trn_config["additive_gaussian_noise"]
            ),

            # Move flow channel to last dim
            T.ReorderAxes(keys=[FWD_FLOW_KEY, BWD_FLOW_KEY], axes=t3zyx_to_tzyx3)
        ])

    def get_val_transforms(self):
        return T.Compose([
            T.EnsureFloat(keys=[IMAGE_KEY, MI_KEY, MF_KEY, FWD_FLOW_KEY, BWD_FLOW_KEY]),

            # Reorder axes for tensors
            T.ReorderAxes(keys=[IMAGE_KEY], axes=xyzt_to_tzyx),
            T.ReorderAxes(keys=[MI_KEY, MF_KEY], axes=xyz_to_zyx),
            T.ReorderAxes(keys=[FWD_FLOW_KEY, BWD_FLOW_KEY], axes=t3xyz_to_t3zyx),

            # Add channel dimension
            T.AddDimAt(keys=[MI_KEY, MF_KEY], axis=0),
            T.AddDimAt(keys=[IMAGE_KEY], axis=1),

            # Move flow channel to last dim
            T.ReorderAxes(keys=[FWD_FLOW_KEY, BWD_FLOW_KEY], axes=t3zyx_to_tzyx3)
        ])
