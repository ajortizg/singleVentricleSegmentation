from monai.data import NibabelWriter
from monai.data import MetaTensor


def write_metatensor_to_nifti(x: MetaTensor, filepath: str, channel_dim: int = 0, squeeze_end_dims: bool = True, resample=False, verbose=False) -> None:
    """
    Write a MetaTensor to a NIfTI file using NibabelWriter.

    Args:
        x (MetaTensor): The tensor to write.
        filepath (str): The file path to write the tensor to.
        channel_dim (int): The dimension of the channel.
        squeeze_end_dims (bool): Whether to squeeze the end dimensions.
        resample (bool): Whether to resample the data using original affine.
        verbose (bool): Whether to print verbose output.
    """
    writer = NibabelWriter()
    writer.set_data_array(x, channel_dim=channel_dim, squeeze_end_dims=squeeze_end_dims)
    writer.set_metadata(x.meta, resample=resample)
    writer.write(filepath, verbose=verbose)
