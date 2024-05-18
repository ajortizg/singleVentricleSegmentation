import os.path as osp
import time
import os


def create_timestamped_dir(out_dir: str, name: str) -> str:
    """
    Create a directory with a timestamped name under the specified output directory.

    Args:
        out_dir (str): The base directory where the new directory will be created.
        name (str): The name prefix for the new directory.

    Returns:
        str: The path of the newly created directory.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = osp.join(out_dir, f'{name}_{timestr}')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def create_subdir(parent_dir: str, subdir: str) -> str:
    """
    Create a subdirectory under the specified parent directory.

    Args:
        save_dir (str): The parent directory where the new subdirectory will be created.
        subdir (str): The name of the new subdirectory.

    Returns:
        str: The path of the newly created subdirectory.
    """
    subdir_path = osp.join(parent_dir, subdir)
    if not osp.exists(subdir_path):
        os.makedirs(subdir_path)
    return subdir_path
