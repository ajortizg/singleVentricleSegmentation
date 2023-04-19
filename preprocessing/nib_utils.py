import nibabel as nib
import os.path as osp


def save_np_to_nifty(x, save_dir, filename, affine, hdr_old, qform, sform, zooms):
    hdr = nib.nifti1.Nifti1Header.from_header(hdr_old)
    hdr.set_data_shape(x.shape)
    # hdr.set_qform(hdr_old.get_qform())
    # hdr.set_sform(hdr_old.get_sform())
    # hdr.set_zooms(hdr_old.get_zooms())
    hdr.set_qform(qform)
    hdr.set_sform(sform)
    hdr.set_zooms(zooms)
    nib_data = nib.Nifti1Image(x, affine=affine, header=hdr)
    nib.save(nib_data, osp.join(save_dir, filename))
