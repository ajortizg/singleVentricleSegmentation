import os.path as osp
from datetime import datetime
from glob import glob
import path_utils
import time
import pandas as pd
import torch

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

        df = pd.DataFrame()
        for i, dir in enumerate(self.filtered_dirs):
            *_, best_train_acc, best_val_acc, best_test_acc, best_e = self.read_checkpoint(dir, 'checkpoint_best.pth')
            *_, final_train_acc, final_val_acc, final_test_acc, final_e = self.read_checkpoint(dir, 'checkpoint_final.pth')

            df_data = pd.DataFrame(index=[i])
            df_data.insert(0, 'experiment', dir)

            metrics_row = pd.DataFrame(
                {'best_epoch': best_e, 'best_train_acc': best_train_acc, 'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc,
                 'final_epoch': final_e, 'final_train_acc': final_train_acc, 'final_val_acc': final_val_acc, 'final_test_acc': final_test_acc},
                index=[i]).round(3)

            df_row = df_data.join([metrics_row])
            df = pd.concat([df_row, df])
        df.to_excel(filepath, index=False)

    def read_checkpoint(self, dir, filename):
        try:
            checkpoint = torch.load(osp.join(dir, filename))
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            train_acc = checkpoint['train_acc']
            val_acc = checkpoint['val_acc']
            test_acc = checkpoint['test_acc']
            e = checkpoint['epoch']
            return (train_loss, val_loss, train_acc, val_acc, test_acc, e)
        except FileNotFoundError:
            return (0,) * 6

    def from_str(self):
        return self.from_dt.strftime(str_fmt)

    def to_str(self):
        return self.to_dt.strftime(str_fmt)


if __name__ == "__main__":
    search_dir = 'results_george/CNN_*'
    from_dt = datetime.strptime(fmt_date(y='2021', m='04', d='19', hr='00', min='00', seg='00'), str_fmt)
    to_dt = datetime.strptime(fmt_date(y='2023', m='05', d='19', hr='00', min='00', seg='00'), str_fmt)

    report_dir = path_utils.create_sub_dir('results', 'reports')
    print('Creating report from: ', from_dt, ' to: ', to_dt)

    report = Report(from_dt, to_dt)
    n = report.filter(search_dir)
    print(f'Found {n} dirs')

    filepath = osp.join(report_dir, f'chkpt_report_{time.strftime(str_fmt)}.xlsx')
    report.create(filepath)
    print(f'Created report: {filepath}')
