from configparser import ConfigParser
import os.path as osp
from tqdm import tqdm
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import utilities.plots as plots
from utilities import path_utils, stuff
import utilities.transforms.unary_transforms as T1
from segmentation.transforms import one_hot
from cnn.dataset import SingleVentricleDataset, DatasetMode, LoadFlowMode
from datasets.preprocessing import viz

if __name__ == "__main__":
    config = ConfigParser()
    config.read('parser/flow_warping.ini')
    data = config["DATA"]
    params = config["PARAMETERS"]
    n_classes = params.getint("n_classes")
    save_size = (params.getint('save_nz'), params.getint('save_ny'), params.getint('save_nx'))

    img4d_transf = T1.Compose([T1.ToTensor()])
    mask_transf = T1.Compose([T1.ToTensor()])

    if params["dataset_mode"] == "split":
        train_ds = SingleVentricleDataset(config, DatasetMode.TRAIN, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)
        val_ds = SingleVentricleDataset(config, DatasetMode.VAL, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)
        test_ds = SingleVentricleDataset(config, DatasetMode.TEST, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf, test_masks_transforms=mask_transf)
        datasets = [train_ds, val_ds, test_ds]
        items = len(train_ds) + len(val_ds) + len(test_ds)
    elif params["dataset_mode"] == "full":
        full_ds = SingleVentricleDataset(config, DatasetMode.FULL, LoadFlowMode.NO_LOAD, img4d_transf, mask_transf)
        datasets = [full_ds]
        items = len(full_ds)
    else:
        raise ValueError("Unknow dataset_mode")

    save_dir = path_utils.create_save_dir(data['output_path'], 'ImageMask')
    stuff.save_config(config, save_dir, 'config.ini')

    mask_posp = T1.Compose([T1.Resize(save_size), T1.Round_V2()])
    img_posp = T1.Compose([T1.Resize(save_size), T1.Normalize()])
    stuff.seeding(42)
    m0_colors = plots.random_colors(n_classes)
    stuff.seeding(19)
    mk_colors = plots.random_colors(n_classes)

    pbar = tqdm(total=items)
    for ds in datasets: 
        for (pname, data, m0, mk, masks, init_ts, final_ts, _, _) in ds:
            if masks is not None:
                print(pname, 'full_cycle')
                timesteps = masks.shape[-1]
            else:
                timesteps = data.shape[-1]

            for t in range(timesteps):
                u = img_posp(data[..., t])
                patient_dir = path_utils.create_sub_dir(save_dir, pname)

                if t == init_ts:
                    mask = masks[..., t] if masks is not None else m0
                    m0 = one_hot(mask_posp(m0).unsqueeze(0).unsqueeze(0), n_classes + 1).squeeze(0)
                    mask = one_hot(mask_posp(mask).unsqueeze(0).unsqueeze(0), n_classes + 1).squeeze(0)
                    mask = T1.ErodeOneHot()(mask)
                    plots.save_img_masks_one_hot(u, [m0, mask], f"{pname}_t{t}.png", patient_dir, [0.2, 1.0], [m0_colors, m0_colors])
                elif t == final_ts:
                    mask = masks[..., t] if masks is not None else mk
                    mk = one_hot(mask_posp(mk).unsqueeze(0).unsqueeze(0), n_classes + 1).squeeze(0)
                    mask = one_hot(mask_posp(mask).unsqueeze(0).unsqueeze(0), n_classes + 1).squeeze(0)
                    mask = T1.ErodeOneHot()(mask)
                    plots.save_img_masks_one_hot(u, [mk, mask], f"{pname}_t{t}.png", patient_dir, [0.2, 1.0], [m0_colors, m0_colors])
                else:
                    if masks is not None:
                        mask = one_hot(mask_posp(mask).unsqueeze(0).unsqueeze(0), n_classes + 1).squeeze(0)
                        mask = T1.ErodeOneHot()(mask)
                        plots.save_img_masks_one_hot(u, [mask], f"{pname}_t{t}.png", patient_dir, [1.0], [m0_colors])
                    else:
                        plots.save_img_masks_one_hot(u, None, f'{pname}_t{t}.png', patient_dir, alphas=None, colors=None)

            viz.save_gif(patient_dir, 150)

            pbar.update(1)
