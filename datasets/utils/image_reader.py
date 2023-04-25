import numpy as np
import nibabel as nib

class ImageReader:
    def __init__(self, ext):
        self.ext = ext

    def __call__(self,file):
        if self.ext == '.nii.gz':
            x_nib =  nib.load(file)
            data = {'data': x_nib.get_fdata(), 'meta' : {'affine': x_nib.affine}}
        elif self.ext == '.npy':
            data = {'data': np.load(file), 'meta' : None}
        return data
        