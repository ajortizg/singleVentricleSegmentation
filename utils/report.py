import configparser
import os.path as osp
from datetime import datetime
from glob import glob
import plots
import time
import pandas as pd
import torch
import argparse
from enum import Enum


class ReportMode(str, Enum):
    CNN = 'CNN_*'
    FT = 'FT_*'


str_fmt = '%Y%m%d-%H%M%S'


class Report:
    def __init__(self, from_dt, to_dt, mode):
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.filtered_dirs = []
        self.mode = mode

    def filter(self, root_dir):
        cnn_dirs = sorted(glob(osp.join(root_dir, self.mode)), reverse=True)
        for exp_dir in cnn_dirs:
            dt = datetime.strptime(exp_dir.split(osp.sep)[-1].split('_')[-1], str_fmt)
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

            metrics_row = pd.DataFrame({'best_val_loss': best_val_loss, 'best_val_acc': best_val_acc,
                                        'train_loss': train_loss, 'train_acc': train_acc,
                                        'val_loss': val_loss, 'val_acc': val_acc, 'test_acc': test_acc}, index=[i]).round(3)

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
        if self.mode == ReportMode.CNN:
            data_items.pop('volumes_subdir_path')
            data_items.pop('segmentations_subdir_path')
            data_items.pop('segmentations_file_name')
            data_items.pop('output_path')
            data_items['base_path_3d'] = data_items['base_path_3d'].split(osp.sep)[-1]
        return data_items

    def from_str(self):
        return self.from_dt.strftime(str_fmt)

    def to_str(self):
        return self.to_dt.strftime(str_fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_date', help='create report from date')
    parser.add_argument('--to_date', help='create report to date')
    parser.add_argument('--mode', help='cnn or ft', default='cnn')
    args = parser.parse_args()

    save_dir = 'results'
    report_dir = plots.createSubDirectory(save_dir, 'reports')

    if args.mode == 'cnn':
        mode = ReportMode.CNN
    else:
        mode = ReportMode.FT

    from_dt = datetime.strptime(args.from_date, str_fmt)
    to_dt = datetime.strptime(args.to_date, str_fmt)

    print('Creating report from: ', from_dt, ' to: ', to_dt)
    print('Mode: ', args.mode)

    # from_dt = datetime(year=2022, month=8, day=17, hour=17, minute=0)
    # to_dt = datetime(year=2022, month=8, day=17, hour=23, minute=59)

    report = Report(from_dt, to_dt, mode)
    report.filter('results')
    report.create(report_dir, f'report_{time.strftime(str_fmt)}.xlsx')
