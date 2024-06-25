import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def main(task, size='full', oversample=False):
    with open(f'./data/{task}/mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)

    n_keys = list(filter(lambda x: x.split('_')[1][0] == 'n', mapping.keys()))
    p_keys = list(filter(lambda x: x.split('_')[1][0] == 'p', mapping.keys()))
    total = len(n_keys) + len(p_keys)

    if size == '2k':
        ratio = 2000 / total
    elif size == '4k':
        ratio = 4000 / total
    else:
        ratio = 1

    num_n = int(ratio * len(n_keys))
    num_p = int(ratio * len(p_keys))
    n_keys = np.random.choice(n_keys, num_n, replace=False)
    p_keys = np.random.choice(p_keys, num_p, replace=False)

    train_n, val_n = train_test_split(n_keys, test_size=0.2, random_state=None)
    train_p, val_p = train_test_split(p_keys, test_size=0.2, random_state=None)

    if oversample:
        orig = len(train_p)
        multiplier = int(len(train_n) / len(train_p))   # +ve is always the minority class
        train_p = train_p.repeat(multiplier)
        print(f'Oversampled positive class ({orig} -> {len(train_p)})')

    train_keys = list(np.hstack([train_n, train_p]))
    val_keys = list(np.hstack([val_n, val_p]))

    np.random.shuffle(train_keys)
    np.random.shuffle(val_keys)

    print(f'Train: {len(train_p)} (+ve), {len(train_n)} (-ve)')
    print(f'Val: {len(val_p)} (+ve), {len(val_n)} (-ve)')

    os.makedirs(f'./data/{task}/splits/{size}', exist_ok=True)

    with open(f'./data/{task}/splits/{size}/train.pkl', 'wb') as f:
        pickle.dump(train_keys, f)

    with open(f'./data/{task}/splits/{size}/val.pkl', 'wb') as f:
        pickle.dump(val_keys, f)
