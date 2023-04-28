import nibabel as nib
import os.path as osp
import numpy as np


def save_np_to_nifty(x, save_dir, filename, hdr_old=None, zooms=None):
    if hdr_old is not None:
        hdr = nib.nifti1.Nifti1Header()
        hdr.set_data_shape(x.shape)
        hdr.set_qform(hdr_old.get_qform())
        hdr.set_sform(hdr_old.get_sform())
        hdr.set_zooms(zooms) if zooms is not None else hdr_old.get_zooms()
    else:
        hdr = None
    affine = np.diag([1, 1, 1, 1])
    nib_data = nib.Nifti1Image(x, affine=affine, header=hdr)
    nib.save(nib_data, osp.join(save_dir, filename))
