from dataset import SingleVentricleDataset, DatasetMode
import configparser
import os.path as osp
import sys
from tqdm import tqdm
import pandas as pd
import transforms as T

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from utils import plots


def save_patient_data(vol, ms, md, ts, td,
                       save_dir_vol, save_dir_masks, patient_name):
    patient_mask_dir = plots.createSubDirectory(save_dir_masks, patient_name)
    plots.save4D_torch_to_nifty(vol, save_dir_vol, patient_name + '.nii.gz')
    plots.save3D_torch_to_nifty(ms, patient_mask_dir, patient_name + '_Systole_Labelmap')
    plots.save3D_torch_to_nifty(md, patient_mask_dir, patient_name + '_Diastole_Labelmap')
    row = {'Name': patient_name, 'Systole': ts, 'Diastole': td}
    df = pd.DataFrame(row, index=[0])
    return df


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('parser/configCNN.ini')

    transf = T.Compose([
        T.RandomFlipZ(p=0.6),
        T.RandomFlipY(p=0.6),
        T.RandomFlipX(p=0.6),
        T.RandomRotate(p=1.0, range_x=(-10, 10))])

    ds = SingleVentricleDataset(config, DatasetMode.FULL, None, load_flow=True)
    save_dir = plots.createSaveDirectory(config.get('DATA', 'OUTPUT_PATH'), 'DA')
    save_dir_vol = plots.createSubDirectory(save_dir, ds.volumes_subdir_path)
    save_dir_masks = plots.createSubDirectory(save_dir, ds.segmentations_subdir_path)

    # save config file to save directory
    conifg_output = osp.sep.join([save_dir, "config.ini"])
    with open(conifg_output, 'w') as configfile:
        config.write(configfile)

    N = 10
    df = pd.DataFrame(columns=['Name', 'Systole', 'Diastole'])
    pbar = tqdm(total=len(ds))
    for (patient_name, vol, *_) in ds:
        pbar.set_postfix_str(f'P: {patient_name}')
        NZ, NY, NX, NT = vol.shape
        ms, md = ds.systole_diastole_mask(patient_name)
        ts, td = ds.systole_diastole_time(patient_name)

        # Save original data
        df_patient = save_patient_data(vol, ms, md, ts, td, save_dir_vol, save_dir_masks, patient_name)
        df = pd.concat([df, df_patient], ignore_index=True)

        # Generate new data
        for i in range(N):
            patiente_name_t = patient_name + f'_A_{i}'
            vol_t, ms_t, md_t = transf(vol, ms, md)
            df_patient = save_patient_data(vol_t, T.Round(th=0.5)(ms_t), T.Round(th=0.5)(md_t),
                                            ts, td, save_dir_vol, save_dir_masks, patiente_name_t)
            df = pd.concat([df, df_patient], ignore_index=True)

        pbar.update(1)

    df.to_excel(osp.join(save_dir, ds.segmetations_filename))
