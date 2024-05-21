import time
from absl import flags, app
from ml_collections import config_dict, config_flags
import yaml
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np

from svs.modules.datasets import MRIBaseDataset, xyzt_to_tzyx
from svs.utils import dirs, flow_utils
from svs.modules.tvl1_flow import TVL13DOpticalFlow

# Load default configuration from config file. However the config parameters can be modified
# via command line args, for example: python scripts/flow.py --config.flow.mode=backward
_CONFIG = config_flags.DEFINE_config_dict('config', config_dict.ConfigDict(
    yaml.load(open(osp.join('conf', 'flow.yaml'), 'r'), Loader=yaml.FullLoader)))


def main(_):
    cfg = _CONFIG.value

    save_dir = dirs.create_timestamped_dir(cfg.data.out_dir, f'flow-{cfg.flow.mode}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'TVL1-3D Optical flow\nSave dir: {save_dir}\nMode: {cfg.flow.mode}\nDevice: {device}')

    assert device.type == 'cuda', "Optical flow computation requires a CUDA device."

    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    optflow = TVL13DOpticalFlow(
        cfg.flow.num_scales,
        cfg.flow.max_warps,
        cfg.flow.max_outer_iterations,
        cfg.flow.weight_matching,
        cfg.flow.weight_tv,
        cfg.flow.primal_dual.type,
        cfg.flow.primal_dual.params,
        cfg.flow.anisotropic_diff.enabled,
        cfg.flow.anisotropic_diff.params,
        cfg.flow.gaussian_blur.enabled,
        cfg.flow.gaussian_blur.sigma,
        cfg.flow.interpolation.type,
        cfg.flow.interpolation.boundary,
        cfg.flow.mesh.type,
        cfg.flow.mesh.length,
        device
    )

    pbar = tqdm(total=len(ds))
    for i in range(len(ds)):
        img = torch.from_numpy(ds[i].img_array().transpose(xyzt_to_tzyx)).float().to(device)
        patient_name = ds[i].name

        *_, indices = flow_utils.compute_timepoints_and_indices(ds[i].tdia, ds[i].tsys, None, None, cfg.flow.mode)

        # Initialization of optical flow and mask
        NT, NZ, NY, NX = img.shape
        u = torch.zeros((NZ, NY, NX, 3), dtype=torch.float32, device=device)
        us = torch.zeros((len(indices) - 1, 3, NX, NY, NZ), dtype=torch.float32, device=device)
        p = torch.zeros((NZ, NY, NX, 3, 3), dtype=torch.float32, device=device)

        tic = time.time()
        for j in range(len(indices) - 1):
            I0 = img[indices[j+1]]
            I1 = img[indices[j]]
            pbar.set_postfix_str(f'P: {ds[i].name}, ({indices[j]}->{indices[j+1]}/{indices[-1]})')

            u, p = optflow.compute(I0, I1, u, p)
            if cfg.flow.median_filter.enabled:
                flow_utils.apply_median_filter(u, cfg.flow.median_filter.kernel, device)
            us[i] = u.permute(3, 2, 1, 0)  # change to (3,x,y,z)

        # Save flow with shape (t,3,x,y,z)
        np.save(osp.join(save_dir, f'{patient_name}_{cfg.flow.mode}_flow.npy'), us.cpu().detach().numpy())
        print(f'{patient_name}: {(time.time() - tic) / 60.0}')
        pbar.update(1)

    with open(osp.join(save_dir, "config.json"), "w") as f:
        f.write(cfg.to_json(indent=4))


if __name__ == '__main__':
    app.run(main)
