import configparser
import os.path as osp
from datetime import datetime
from glob import glob
import plots
import time
import pandas as pd


class TrainingReport:
    def __init__(self, from_dt, to_dt):
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.str_fmt = '%Y%m%d-%H%M%S'
        self.filtered_dirs = []

    def filter(self, root_dir):
        experiment_dirs = sorted(glob(osp.join(root_dir, 'CNN_*')), reverse=True)
        for exp_dir in experiment_dirs:
            dt = datetime.strptime(exp_dir.split(osp.sep)[-1].split('_')[-1], self.str_fmt)
            if dt >= self.from_dt and dt <= self.to_dt:
                self.filtered_dirs.append(exp_dir)

    def create(self, save_dir, filename):
        if len(self.filtered_dirs) == 0:
            print('filter files first')
            return

        df = pd.DataFrame()
        for i, dir in enumerate(self.filtered_dirs):
            config = configparser.ConfigParser()
            config.read(osp.join(dir, 'config.ini'))

            data_items = self.clean_data_dict(dict(config.items('DATA')))
            param_items = dict(config.items('PARAMETERS'))
            da_items = dict(config.items('DATA_AUGMENTATION'))
            warping_items = dict(config.items('WARPING'))

            df_data = pd.DataFrame(data_items, index=[i])
            df_data.insert(0, 'experiment', dir.split(osp.sep)[-1])
            df_param = pd.DataFrame(param_items, index=[i])
            df_warping = pd.DataFrame(warping_items, index=[i])
            df_da = pd.DataFrame(da_items, index=[i])

            df_row = df_data.join([df_param, df_warping, df_da])
            df = pd.concat([df_row, df])
        df.to_excel(osp.join(save_dir, filename), index=False)

    def clean_data_dict(self, data_items):
        data_items.pop('volumes_subdir_path')
        data_items.pop('segmentations_subdir_path')
        data_items.pop('segmentations_file_name')
        data_items.pop('output_path')
        data_items['base_path_3d'] = data_items['base_path_3d'].split(osp.sep)[-1]
        return data_items

    def from_str(self):
        return self.from_dt.strftime(self.str_fmt)

    def to_str(self):
        return self.to_dt.strftime(self.str_fmt)


if __name__ == "__main__":
    root_dir = 'results'
    report_dir = plots.createSubDirectory(root_dir, 'reports')

    from_dt = datetime(year=2022, month=8, day=9, hour=0, minute=0)
    to_dt = datetime(year=2022, month=8, day=11, hour=0, minute=0)

    report = TrainingReport(from_dt, to_dt)
    report.filter('results')
    report.create(report_dir, f'report_{time.strftime(report.str_fmt)}.xlsx')
