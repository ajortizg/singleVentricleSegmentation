from svs.preprocessing import setup_acdc
from svs.preprocessing import cutting
from svs.preprocessing import prolongation
from svs.preprocessing import normalization
from svs.preprocessing import split
from svs.preprocessing import orientation
from svs.preprocessing import isotropic_resample


def run():
    """
    Execute the preprocessing pipeline for the ACDC dataset.

    This function runs a sequence of preprocessing steps: setting up the ACDC dataset,
    cutting the images, prolonging the images, normalizing the intensities, and
    splitting the dataset into training, validation, and test sets.

    It collects and prints the directory paths of the intermediate results for each step.
    """
    tree_dirs = []
    save_dir = setup_acdc.run()
    tree_dirs.append(save_dir)

    save_dir = orientation.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = cutting.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = isotropic_resample.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = prolongation.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = normalization.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = split.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    for i, d in enumerate(tree_dirs):
        print(f'step {i}: {d}')


if __name__ == '__main__':
    run()
