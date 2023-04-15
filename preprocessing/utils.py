import os.path as osp

import nibabel as nib


def save_np_to_nifty(x, save_dir, filename, affine, hdr_old):
    hdr = nib.nifti1.Nifti1Header.from_header(hdr_old)
    hdr.set_data_shape(x.shape)
    hdr.set_qform(hdr_old.get_qform())
    hdr.set_sform(hdr_old.get_sform())
    hdr.set_zooms(hdr_old.get_zooms()[:-1])
    nii_data = nib.Nifti1Image(x, affine=affine, header=hdr)
    nib.save(nii_data, osp.join(save_dir, filename))
