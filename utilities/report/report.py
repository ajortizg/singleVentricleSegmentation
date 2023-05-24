from configparser import ConfigParser
import os.path as osp
from datetime import datetime
from glob import glob
import time
import pandas as pd
import torch

import path_utils

str_fmt = '%Y%m%d-%H%M%S'


def fmt_date(y, m, d, hr, min, seg):
    return y + m + d + '-' + hr + min + seg


class Report:
    def __init__(self, from_dt, to_dt):
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.filtered_dirs = []

    def filter(self, search_dir):
        cnn_dirs = sorted(glob(search_dir), reverse=True)
        for exp_dir in cnn_dirs:
            dt = datetime.strptime(exp_dir.split(osp.sep)[-1].split('_')[-1], str_fmt)
            if dt >= self.from_dt and dt <= self.to_dt:
                self.filtered_dirs.append(exp_dir)
        return len(self.filtered_dirs)

    def create(self, filepath):
        if len(self.filtered_dirs) == 0:
            print('filter files first')
            return

        report_df = pd.DataFrame()
        for i, dir in enumerate(self.filtered_dirs):
            config = ConfigParser()
            config.read(osp.join(dir, 'config.ini'))

            experiment_df = pd.DataFrame()
            for section in config.sections():
                items_df = pd.DataFrame({name: val for (name, val) in config.items(section)}, index=[i])
                experiment_df = pd.concat([experiment_df, items_df], axis=1)
            experiment_df.insert(0, 'experiment', dir.split(osp.sep)[-1])

            metrics_df = pd.DataFrame()
            for metric, flag in zip([self.read_metrics(dir, 'checkpoint_best.pth'), 
                                     self.read_metrics(dir, 'checkpoint_final.pth')],
                                    ['best', 'final']):
                df = pd.DataFrame({flag + '_' + k: v for (k, v) in metric.items()}, index=[i]).round(3)
                metrics_df = pd.concat([metrics_df, df], axis=1)

            experiment_df = pd.concat([experiment_df, metrics_df], axis=1)
            report_df = pd.concat([experiment_df, report_df], axis=0)
        report_df.to_excel(filepath, index=False)

    def read_metrics(self, dir, filename):
        metrics = {}
        try:
            checkpoint = torch.load(osp.join(dir, filename))
            # metrics['train_loss'] = checkpoint['train_loss']
            # metrics['val_loss'] = checkpoint['val_loss']
            metrics['train_acc'] = checkpoint['train_acc']
            metrics['val_acc'] = checkpoint['val_acc']
            metrics['test_acc'] = checkpoint['test_acc']
            metrics['epoch'] = checkpoint['epoch']
        except FileNotFoundError:
            pass
        return metrics

    def from_str(self):
        return self.from_dt.strftime(str_fmt)

    def to_str(self):
        return self.to_dt.strftime(str_fmt)


if __name__ == "__main__":
    search_dir = 'results/cnn/singleVentricleData_split/CNN_*'
    from_dt = datetime.strptime(fmt_date(y='2022', m='04', d='18', hr='00', min='00', seg='00'), str_fmt)
    to_dt = datetime.strptime(fmt_date(y='2024', m='05', d='19', hr='00', min='00', seg='00'), str_fmt)

    report_dir = path_utils.create_sub_dir('results', 'reports')
    print('Creating report from: ', from_dt, ' to: ', to_dt)

    report = Report(from_dt, to_dt)
    n = report.filter(search_dir)
    print(f'Found {n} dirs')

    filepath = osp.join(report_dir, f'report_{time.strftime(str_fmt)}.xlsx')
    report.create(filepath)
    print(f'Created report: {filepath}')
