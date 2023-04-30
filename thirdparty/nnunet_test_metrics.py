import os
import sys
import os.path as osp
import nibabel as nib
from glob import glob
import numpy as np
import json
from collections import defaultdict
import torch
from monai.metrics.meandice import compute_dice
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../'))
sys.path.append(ROOT_DIR)
import segmentation.transforms as T


pred_labels_dir = 'results/nnUNet_prediction'
true_labels_dir = 'data/nnUNet_raw/Dataset013_SVDraw/labelsTs'

pred_files = sorted(glob(osp.join(pred_labels_dir, '*.nii.gz')))
true_files = sorted(glob(osp.join(true_labels_dir, '*.nii.gz')))
f = open('data/nnUNet_raw/Dataset013_SVDraw/test.json', mode='r')
data_json = json.load(f)

preds_per_patient = defaultdict(list)
trues_per_patient = defaultdict(list)

for predf, truef in zip(pred_files, true_files):
    meta_pred_patient = data_json[osp.basename(predf)]
    preds_per_patient[meta_pred_patient['patient']].append({
        'file': predf,
        'es': meta_pred_patient['es'],
        'ed': meta_pred_patient['ed']
    })

    meta_true_patient = data_json[osp.basename(truef)]
    trues_per_patient[meta_true_patient['patient']].append({
        'file': truef,
        'es': meta_true_patient['es'],
        'ed': meta_true_patient['ed']
    })


patients = list(preds_per_patient.keys())
dice_per_patient = {}
hd_per_patient = {}

transforms = T.Compose([
    T.AddNLeadingDims(2, ['pred', 'true']),
    T.BCXYZ_To_BCZYX(['pred', 'true']),
    T.OneHotEncoding(2, ['pred', 'true']),
    T.ToTensor(['pred', 'true'])
])

bounds_per_patient = {}
for patient in patients:
    es = preds_per_patient[patient][0]['es']
    ed = preds_per_patient[patient][0]['ed']
    bounds_per_patient[patient] = {'ti': min(es, ed), 'tf': max(es, ed)}

for patient in patients:
    list_preds = preds_per_patient[patient]
    list_trues = trues_per_patient[patient]
    assert len(list_preds) == len(list_trues)

    dices, hds = [], []
    for pred_dict, true_dict in zip(list_preds, list_trues):
        predf = pred_dict['file']
        truef = true_dict['file']
        data = {}
        data['pred'] = nib.load(predf).get_fdata()
        data['true'] = nib.load(truef).get_fdata()
        data = transforms(data)

        dice = compute_dice(data['pred'], data['true'], include_background=False).mean().item()
        hd = compute_hausdorff_distance(data['pred'], data['true'], include_background=False).mean().item()
        dices.append(dice)
        hds.append(hd)

    dice_per_patient[patient] = np.array(dices)
    hd_per_patient[patient] = np.array(hds)

print('Accuracy for ED-ES')
for patient in patients:
    ti = bounds_per_patient[patient]['ti']
    tf = bounds_per_patient[patient]['tf']

    print(patient)
    print('\tDice: ', dice_per_patient[patient][ti:tf].mean())
    print('\tHD: ', hd_per_patient[patient][ti:tf].mean())
