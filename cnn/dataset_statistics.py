import torch
from dataset import SingleVentricleDataset, DatasetMode
import configparser
from tqdm import tqdm
import sys


config = configparser.ConfigParser()
config.read('parser/configCNN.ini')

ds = SingleVentricleDataset(config, DatasetMode.FULL, load_flow=False)
pbar = tqdm(total=len(ds))
sum = 0
squared_sum = 0
for idx in range(len(ds)):
    (pname, data, m0, mk, init_ts, final_ts, _, _) = ds[idx]
    pbar.set_postfix_str(f'P: {pname}')
    sum += data.mean()
    squared_sum += torch.mean(data**2)
    pbar.update(1)

mean = sum / len(ds)
std = (squared_sum / len(ds) - mean**2)**0.5
print(mean, std)
