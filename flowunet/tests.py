import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
from datasets.flowunet_dataset import FlowUNetDataset

if __name__ == '__main__':
    ds = FlowUNetDataset('results/preprocessed/singleVentricleData', 'full', None, load_flow=True)

    for data in ds:
        img = data['image']
        label = data['label']
        es = data['es']
        ed = data['ed']
        patient = data['patient']
        ff = data['forward_flow']
        bf = data['backward_flow']

        print(patient, es, ed)
        print(img.shape)
        print(label.shape)
        print(ff.shape)
        print(bf.shape)
        print('=' * 80)
