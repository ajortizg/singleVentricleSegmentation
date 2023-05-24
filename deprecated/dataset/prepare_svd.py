import monai
from monai.config import KeysCollection
from monai.utils import ensure_tuple_rep
import monai.transforms as T
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import sys
import torch.nn.functional as F
import shutil
import math
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from utilities import path_utils

bad_patients_1 = {"Adolescent_7",
                  "Adolescent_26",
                  "Adolescent_54",
                  "Adolescent_75",
                  "Child_27",
                  "Adult_6",
                  "Adult_11",
                  "Adult_16",
                  "Adult_17"}

bad_patients_2 = {"Adolescent_86",
                  "Child_2",
                  "Child_24",
                  "Child_26",
                  "Child_43",
                  "Adult_7",
                  "Adult_36",
                  "Adult_37",
                  "Child_50",
                  "Child_69",
                  "Child_72",
                  "Adult_62",
                  "Adult_85"}


def save_slice(data, z, squeeze_first_dim, filename, alpha=0.2):
    plt.figure("visualize", (8, 4))
    plt.subplot(1, 2, 1)
    plt.title("es")
    plt.imshow(data["image"][data['es'], ..., z], cmap="gray")
    label_es = data["label_es"].squeeze(0) if squeeze_first_dim else data["label_es"]
    plt.imshow(label_es[..., z], cmap="jet", alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.title("ed")
    plt.imshow(data["image"][data['ed'], ..., z], cmap="gray")
    label_ed = data["label_ed"].squeeze(0) if squeeze_first_dim else data["label_ed"]
    plt.imshow(label_ed[..., z], cmap="jet", alpha=alpha)
    plt.savefig(filename, dpi=100)


def save_test_slice(data, test_data, t, z, squeeze_first_dim, filename, alpha=0.2):
    plt.figure("visualize_test", (4, 4))
    plt.imshow(data["image"][t, ..., z], cmap="gray")
    label = test_data["label"].squeeze(0) if squeeze_first_dim else data["label"]
    plt.imshow(label[..., z], cmap="jet", alpha=alpha)
    plt.savefig(filename, dpi=100)


def mask_for_cropping(data, xtol=10, ytol=10, ztol=10):
    mask_or = torch.bitwise_or(data["label_es"] > 0, data["label_ed"] > 0)
    _, nx, ny, nz = mask_or.shape
    _, x, y, z = torch.nonzero(mask_or, as_tuple=True)
    xmin = torch.min(x)
    xmax = torch.max(x)
    ymin = torch.min(y)
    ymax = torch.max(y)
    zmin = torch.min(z)
    zmax = torch.max(z)
    # Extend range by tolerance
    xmin_total = max(0, xmin - xtol)
    xmax_total = min(nx - 1, xmax + xtol)
    ymin_total = max(0, ymin - ytol)
    ymax_total = min(ny - 1, ymax + ytol)
    zmin_total = max(0, zmin - ztol)
    zmax_total = min(nz - 1, zmax + ztol)
    mask_or[:, xmin_total:xmax_total + 1, ymin_total:ymax_total + 1, zmin_total:zmax_total + 1] = 1
    return mask_or


def percentile(x: torch.Tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (x.numel() - 1))
    result = x.view(-1).kthvalue(k).values.item()
    return result


class QuadraticNormalizationd(T.transform.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            per95 = percentile(d[key], 95)
            d[key] = d[key].clip(0, per95)
            ed_avg = float(np.mean(d[key][d["ed"]], where=d["label_ed"].squeeze(0) > 0))
            es_avg = float(np.mean(d[key][d["es"]], where=d["label_es"].squeeze(0) > 0))
            avg = 0.5 * (ed_avg + es_avg)
            # n(I) = a I/sqrt(1+beta I**2)
            norm_a = math.sqrt(per95 * per95 - avg * avg) / (math.sqrt(3) * per95 * avg)
            norm_b = (per95 * per95 - 4. * avg * avg) / (3. * per95 * per95 * avg * avg)
            print(type(norm_a), type(norm_b))
            d[key] = norm_a * d[key] / torch.sqrt(1 + norm_b * d[key]**2)
            print("norm(per95) = ", norm_a * per95 / math.sqrt(1 + norm_b * per95 * per95))
            print("norm(avg) = ", norm_a * avg / math.sqrt(1 + norm_b * avg * avg))
            print(torch.aminmax(d[key]))
        return d


if __name__ == "__main__":
    # root_dir = "data/singleVentricleData/"
    root_dir = "results/ACDCData_20230522-085510"
    img_ext = ".nii.gz"
    label_ext = ".nii.gz"
    images_dir = "NIFTI_4D_Datasets"
    segmentations_dir = "NIFTI_Single_Ventricle_Segmentations"
    df = pd.read_excel(osp.join(root_dir, "Segmentation_volumes.xlsx"))

    train_transforms = T.Compose([
        T.LoadImaged(keys=("image", "label_es", "label_ed"), image_only=False),
        T.EnsureChannelFirstd(keys=("image", "label_es", "label_ed")),
        # T.Orientationd(keys=("image", "label_es", "label_ed"), axcodes="PIL"),
        T.Spacingd(keys=("image", "label_es", "label_ed"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest"))
    ])

    test_transforms = T.Compose([
        T.LoadImaged(keys=('label'), image_only=False),
        T.EnsureChannelFirstd(keys=("label")),
        # T.Orientationd(keys=("label"), axcodes="PIL"),
        T.Spacingd(keys=("label"), pixdim=(1.0, 1.0, 1.0), mode=("nearest"))
    ])

    save_dir = path_utils.create_save_dir('results', 'acdc_iso')
    print('Save dir: ', save_dir)

    for i in tqdm(range(len(df))):
        patient = df.loc[i, "Name"]
        es = df.loc[i, "Systole"]
        ed = df.loc[i, "Diastole"]
        test = df.loc[i, "Full"]

        data_dirs = {
            "image": osp.join(root_dir, osp.join(images_dir, f"{patient}{img_ext}")),
            "label_es": osp.join(root_dir, osp.join(segmentations_dir, patient, f"{patient}_Systole_Labelmap{label_ext}")),
            "label_ed": osp.join(root_dir, osp.join(segmentations_dir, patient, f"{patient}_Diastole_Labelmap{label_ext}")),
            "es": es,
            "ed": ed
        }
        data = train_transforms(data_dirs)

        # *******************************************
        data["label_or"] = mask_for_cropping(data)
        crop = T.CropForegroundd(keys=("image", "label_es", "label_ed"), source_key="label_or")
        resize = T.Resized(keys=("image", "label_es", "label_ed"), spatial_size=(80, 80, 80), mode=("trilinear", "nearest", "nearest"))
        orig_nt = data["image"].shape[0]
        difft = 40 - orig_nt
        padder = T.Padd(keys=("image"), padder=T.Pad(to_pad=[(0, difft), (0, 0), (0, 0), (0, 0)]))
        norm = QuadraticNormalizationd(keys=("image"))

        data = crop(data)
        data = resize(data)
        data = padder(data)
        print(data["image"].pixdim)
        data = norm(data)
        print(data["image"].pixdim)
        # *******************************************

        if test:
            test_saver = T.SaveImaged(
                keys=("label"),
                output_dir=osp.join(save_dir, segmentations_dir, patient),
                output_postfix="",
                output_ext='.nii.gz',
                resample=False,
                separate_folder=False
            )
            for t in range(orig_nt):
                test_data_dir = {"label": osp.join(root_dir, segmentations_dir, patient, f"{patient}_{t}_Labelmap{label_ext}")}
                test_data = test_transforms(test_data_dir)

                # *******************************************
                test_data["label_or"] = data["label_or"]
                crop = T.CropForegroundd(keys=("label"), source_key="label_or")
                resize = T.Resized(keys=("label"), spatial_size=(80, 80, 80), mode=("nearest"))

                test_data = crop(test_data)
                test_data = resize(test_data)
                # *******************************************

                if patient in bad_patients_1:
                    test_data["label"] = test_data["label"].permute(0, 2, 3, 1)
                if patient in bad_patients_2:
                    test_data["label"] = test_data["label"].permute(0, 1, 3, 2)

                test_saver(test_data)
                save_test_slice(data, test_data, t, 30, True, osp.join(path_utils.create_sub_dir(save_dir, "overlay"), f"{patient}_{t}_trans.png"))

        if patient in bad_patients_1:
            data["image"] = data["image"].permute(0, 2, 3, 1)
            data["label_es"] = data["label_es"].permute(0, 2, 3, 1)
            data["label_ed"] = data["label_ed"].permute(0, 2, 3, 1)

        if patient in bad_patients_2:
            data["image"] = data["image"].permute(0, 1, 3, 2)
            data["label_es"] = data["label_es"].permute(0, 1, 3, 2)
            data["label_ed"] = data["label_ed"].permute(0, 1, 3, 2)

        df.loc[i, "orig_NT"] = orig_nt
        df.loc[i, ["pixdim_x", "pixdim_y", "pixdim_z"]] = data["image"].pixdim.tolist()

        img_saver = T.SaveImaged(
            keys=("image"),
            output_dir=osp.join(save_dir, images_dir),
            output_postfix="",
            output_ext='.nii.gz',
            resample=False,
            separate_folder=False
        )
        img_saver(data)

        label_saver = T.SaveImaged(
            keys=("label_es", "label_ed"),
            output_dir=osp.join(save_dir, segmentations_dir, patient),
            output_postfix="",
            output_ext='.nii.gz',
            resample=False,
            separate_folder=False
        )
        label_saver(data)
        save_slice(data, 30, True, osp.join(path_utils.create_sub_dir(save_dir, "overlay"), f"{patient}_trans.png"))

    df.to_excel(osp.join(save_dir, "Segmentation_volumes.xlsx"), index=False)
