from svs.preprocessing import setup_acdc
from svs.preprocessing import cutting
from svs.preprocessing import prolongation


def run():
    tree_dirs = []
    save_dir = setup_acdc.run()
    tree_dirs.append(save_dir)

    save_dir = cutting.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    save_dir = prolongation.run(base_dir=save_dir)
    tree_dirs.append(save_dir)

    for i, d in enumerate(tree_dirs):
        print(f'step {i}: {d}')


if __name__ == '__main__':
    run()
