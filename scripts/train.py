import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from svs.modules.datasets import LitNNDataset, xyz_to_zyx, xyzt_to_tzyx, t3xyz_to_t3zyx, t3zyx_to_tzyx3
import svs.modules.transforms as T
from svs.utils.constants import *
from svs.models.lit_unet import LitUNet


def main():
    transforms = T.Compose([
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

    data = LitNNDataset(
        num_workers=8,
        batch_size=2,
        train_config={
            'base_dir': 'data/refactor_prep/acdc_lv',
            'imgs_dir': 'NIFTI_4D_Datasets',
            'segs_dir': 'NIFTI_Single_Ventricle_Segmentations',
            'metadata_file': 'Segmentation_volumes.xlsx',
            'split': 'train',
            'forward_flow_subdir': 'optical_flow/forward',
            'backward_flow_subdir': 'optical_flow/backward',
            'transforms': transforms
        },
        val_config={
            'base_dir': 'data/refactor_prep/acdc_lv',
            'imgs_dir': 'NIFTI_4D_Datasets',
            'segs_dir': 'NIFTI_Single_Ventricle_Segmentations',
            'metadata_file': 'Segmentation_volumes.xlsx',
            'split': 'val',
            'forward_flow_subdir': 'optical_flow/forward',
            'backward_flow_subdir': 'optical_flow/backward',
            'transforms': transforms
        }
    )

    model = LitUNet(
        epochs=1000,
        lambda_u=1,
        gamma_p=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        model_kwargs=dict(
            num_layers=4,
            num_classes=1,
            input_channels=2,
            features_start=8,
            trilinear=False,
            padding=1,
            kernel_size=(3, 3, 3)
        ),
        warp_kwargs=dict(
            interpolation_type="LINEAR",
            boundary_type="MIRROR",
            mesh_length_type="numDofs",
            lenghts=(1, 1, 1)
        )

    )

    logger = TensorBoardLogger("tb_logs")

    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1000,
        min_epochs=1000,
        num_sanity_val_steps=-1,
        devices=[0],
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
