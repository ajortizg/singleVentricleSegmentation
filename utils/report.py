import configparser
import os.path as osp
from datetime import datetime
from glob import glob
import plots
import time
import pandas as pd
import torch
from enum import Enum


class ReportMode(str, Enum):
    TRAIN = 'CNN_*'
    FT = 'FT_*'


class Report:
    def __init__(self, from_dt, to_dt, mode):
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.str_fmt = '%Y%m%d-%H%M%S'
        self.filtered_dirs = []
        self.mode = mode

    def filter(self, root_dir):
        cnn_dirs = sorted(glob(osp.join(root_dir, self.mode)), reverse=True)
        for exp_dir in cnn_dirs:
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

            best_train_loss, best_train_acc, *_ = self.read_checkpoint(dir, 'best_train_checkpoint.pth')
            *_, best_val_loss, best_val_acc, _ = self.read_checkpoint(dir, 'best_val_checkpoint.pth')
            train_loss, train_acc, val_loss, val_acc, test_acc = self.read_checkpoint(dir, 'checkpoint.pth')

            data_items = self.clean_data_dict(dict(config.items('DATA')))
            param_items = dict(config.items('PARAMETERS'))
            da_items = dict(config.items('DATA_AUGMENTATION'))
            warping_items = dict(config.items('WARPING'))

            df_data = pd.DataFrame(data_items, index=[i])
            df_data.insert(0, 'experiment', dir.split(osp.sep)[-1])
            df_param = pd.DataFrame(param_items, index=[i])
            df_warping = pd.DataFrame(warping_items, index=[i])
            df_da = pd.DataFrame(da_items, index=[i])

            metrics_row = pd.DataFrame({'best_train_loss': f'{best_train_loss:,.3f}', 'best_train_acc': f'{best_train_acc:,.3f}',
                                        'best_val_loss': f'{best_val_loss:,.3f}', 'best_val_acc': f'{best_val_acc:,.3f}',
                                        'train_loss': f'{train_loss:,.3f}', 'train_acc': f'{train_acc:,.3f}',
                                        'val_loss': f'{val_loss:,.3f}', 'val_acc': f'{val_acc:,.3f}', 'test_acc': f'{test_acc:,.3f}'}, index=[i])

            df_row = df_data.join([df_param, df_warping, df_da, metrics_row])
            df = pd.concat([df_row, df])
        df.to_excel(osp.join(save_dir, filename), index=False)

    def read_checkpoint(self, dir, filename):
        try:
            checkpoint = torch.load(osp.join(dir, filename))
            train_loss = checkpoint['train_loss']
            train_acc = checkpoint['train_acc']
            val_loss = checkpoint['val_loss']
            val_acc = checkpoint['val_acc']
            test_acc = checkpoint['test_acc']
            return (train_loss, train_acc, val_loss, val_acc, test_acc)
        except FileNotFoundError:
            return (0, 0, 0, 0, 0)

    def clean_data_dict(self, data_items):
        if self.mode == ReportMode.TRAIN:
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

    mode = ReportMode.TRAIN
    from_dt = datetime(year=2022, month=8, day=12, hour=0, minute=0)
    to_dt = datetime(year=2022, month=8, day=16, hour=23, minute=59)

    report = Report(from_dt, to_dt, mode)
    report.filter('results')
    report.create(report_dir, f'report_{time.strftime(report.str_fmt)}.xlsx')
