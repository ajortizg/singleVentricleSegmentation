from sklearn.model_selection import KFold
import numpy as np


def create_5fold(n, seed=12345):
    splits = []
    indices = np.arange(n)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        train_keys = np.array(indices)[train_idx]
        test_keys = np.array(indices)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits
