import numpy as np
import nibabel as nib

class ImageReader:
    def __init__(self, ext, read_meta=True):
        self.read_meta = read_meta
        self.ext = ext

    def __call__(self, file):
        if self.ext == '.nii.gz':
            x_nib =  nib.load(file)
            data = {'data': x_nib.get_fdata()}
            if self.read_meta:
                data['meta'] = {'affine': x_nib.affine, 
                              'pix_dim': x_nib.header.get_zooms()}
        elif self.ext == '.npy':
            data = {'data': np.load(file)}
            if self.read_meta:
                data['meta'] = None
        return data
        