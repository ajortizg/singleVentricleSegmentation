import numpy as np
from tqdm import tqdm
import os
import torch
from ml_collections import config_dict
import yaml
import os.path as osp
import torch.nn.functional as F


from opticalFlow_cuda_ext import opticalFlow
from utilities import path_utils
from scripts.datasets import MRIBaseDataset, Patient, xyz_to_zyx, zyx_to_xyz, xyzt_to_zyxt, zyxt_to_xyzt


def get_interpolation_type(inter_type_str: str, bound_type_str: str):
    interpolation = None
    if inter_type_str == "NEAREST":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_NEAREST
    elif inter_type_str == "LINEAR":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_LINEAR
    elif inter_type_str == "CUBIC_HERMITESPLINE":
        interpolation = opticalFlow.InterpolationType.INTERPOLATE_CUBIC_HERMITESPLINE
    else:
        raise Exception("Wrong interpolation type in config file")

    boundary = None
    if bound_type_str == "NEAREST":
        boundary = opticalFlow.BoundaryType.BOUNDARY_NEAREST
    elif bound_type_str == "MIRROR":
        boundary = opticalFlow.BoundaryType.BOUNDARY_MIRROR
    elif bound_type_str == "REFLECT":
        boundary = opticalFlow.BoundaryType.BOUNDARY_REFLECT
    else:
        raise Exception("wrong boundary type in configParser")

    return interpolation, boundary


def get_mesh_length(length_type, nz, ny, nx, lz, ly, lx):
    if length_type == "numDofs":
        LZ = nz - 1
        LY = ny - 1
        LX = nx - 1
        return LZ, LY, LX
    elif length_type == "fixed":
        return lz, ly, lx
    else:
        raise Exception("Wrong length type in config file")


def time_pading(maxt: int, img: torch.Tensor) -> torch.Tensor:
    # img torch tensor with shape z,y,x,t
    nt = img.shape[-1]
    diff_t = maxt - nt
    return F.pad(img, [0, diff_t,
                       0, 0,
                       0, 0,
                       0, 0])


def run(config_dir='conf', config_name='preprocessing.yaml', base_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Prolongation, devive: {device}')

    # Load configuration
    cfg = config_dict.ConfigDict(yaml.load(open(osp.join(config_dir, config_name), 'r'), Loader=yaml.FullLoader))

    # Create output dirs
    save_dir = path_utils.create_save_dir(cfg.data.out_dir, "preprocessing_prolongation")
    out_img_dir = path_utils.create_sub_dir(save_dir, cfg.data.imgs_dir)
    out_seg_dir = path_utils.create_sub_dir(save_dir, cfg.data.segs_dir)

    # Prolongation type
    interpolation_type, boundary_type = get_interpolation_type(cfg.prolongation.interpolation, cfg.prolongation.boundary)
    nx_prol = cfg.prolongation.nx
    ny_prol = cfg.prolongation.ny
    nz_prol = cfg.prolongation.nz
    nt_prol = cfg.prolongation.pad_time
    lz_prol, ly_prol, lx_prol = get_mesh_length(cfg.prolongation.length_type, nz_prol, ny_prol, nx_prol,
                                                cfg.prolongation.lz, cfg.prolongation.ly, cfg.prolongation.lx)

    if base_dir is not None:
        cfg.data.base_dir = base_dir
    ds = MRIBaseDataset(cfg.data.base_dir, cfg.data.imgs_dir, cfg.data.segs_dir, cfg.data.metadata_file)

    # Generate columns for prolongation factors
    xprolongfac = np.zeros(len(ds))
    yprolongfac = np.zeros(len(ds))
    zprolongfac = np.zeros(len(ds))
    timeprolongfac = np.zeros(len(ds))

    for i in tqdm(range(len(ds))):
        patient = ds[i]

        zooms = patient.img.header.get_zooms()
        zoom_x = zooms[0]
        zoom_y = zooms[1]
        zoom_z = zooms[2]
        zoom_t = zooms[3]

        # Conver numpy arrays to torch tensor and permute the axes
        img_t = torch.from_numpy(patient.img_array()).float().permute(xyzt_to_zyxt).to(device)
        seg_dia_t = torch.from_numpy(patient.seg_dia_array()).float().permute(xyz_to_zyx).to(device)
        seg_sys_t = torch.from_numpy(patient.seg_sys_array()).float().permute(xyz_to_zyx).to(device)

        # Generate old mesh
        nz, ny, nx, nt = img_t.shape
        lz, ly, lx = get_mesh_length(cfg.prolongation.length_type, nz, ny, nx,
                                     cfg.prolongation.lz, cfg.prolongation.ly, cfg.prolongation.lx)
        mesh_info_old = opticalFlow.MeshInfo3D(nz, ny, nx, lz, ly, lx)

        # generate new mesh for prolongation
        mesh_info_new = opticalFlow.MeshInfo3D(nz_prol, ny_prol, nx_prol, lz_prol, ly_prol, lx_prol)
        prolongation_op = opticalFlow.Prolongation3D(mesh_info_old, mesh_info_new, interpolation_type, boundary_type)

        # Prolongate
        # TODO! Binarize prolongated masks?
        prol_seg_dia_t = prolongation_op.forward(seg_dia_t).round()
        prol_seg_sys_t = prolongation_op.forward(seg_sys_t).round()
        prol_img_t = prolongation_op.forwardVectorField(img_t.contiguous())
        prol_img_t = time_pading(nt_prol, prol_img_t)

        # Create a new patient with prolongated data
        prol_patient = Patient(
            name=patient.name,
            tsys=patient.tsys,
            tdia=patient.tdia,
            init_ts=patient.init_ts,
            final_ts=patient.final_ts
        )
        prol_patient.img_from_array(
            x=prol_img_t.permute(zyxt_to_xyzt).cpu().detach().numpy(),
            affine=patient.img.affine.copy(),
            header=patient.img.header.copy(),
            update_shape=True,
            zooms=(zoom_x * nx / nx_prol, zoom_y * ny / ny_prol, zoom_z * nz / nz_prol, zoom_t / nt_prol)
        )
        prol_patient.seg_dia_from_array(
            x=prol_seg_dia_t.permute(zyx_to_xyz).cpu().detach().numpy(),
            affine=patient.seg_dia.affine.copy(),
            header=patient.seg_dia.header.copy(),
            update_shape=True,
            zooms=(zoom_x * nx / nx_prol, zoom_y * ny / ny_prol, zoom_z * nz / nz_prol)
        )
        prol_patient.seg_sys_from_array(
            x=prol_seg_sys_t.permute(zyx_to_xyz).cpu().detach().numpy(),
            affine=patient.seg_sys.affine.copy(),
            header=patient.seg_sys.header.copy(),
            update_shape=True,
            zooms=(zoom_x * nx / nx_prol, zoom_y * ny / ny_prol, zoom_z * nz / nz_prol)
        )

        prol_patient.save_nifti(out_img_dir, out_seg_dir)

        if cfg.debug.viz:
            prol_patient.viz_data(osp.join(save_dir, "images"), cfg.debug.gif, cfg.debug.dur, aspect_ratio=5.)

        xprolongfac[i] = nx_prol / nx
        yprolongfac[i] = ny_prol / ny
        zprolongfac[i] = nz_prol / nz
        timeprolongfac[i] = nt_prol / nt

    # Save metadata
    output_df = ds.df.copy()
    output_df['xprolongfac'] = xprolongfac
    output_df['yprolongfac'] = yprolongfac
    output_df['zprolongfac'] = zprolongfac
    output_df['timeprolongfac'] = timeprolongfac
    output_df_file = osp.join(save_dir, cfg.data.metadata_file)
    output_df.to_excel(output_df_file, index=False)

    with open(osp.join(save_dir, "config.json"), "w") as f:
        f.write(cfg.to_json(indent=4))
    return save_dir


if __name__ == "__main__":
    run()
