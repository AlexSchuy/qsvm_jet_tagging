import os

import numpy as np
from common import utils, validate
from generation.generate_samples import load_samples
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_train_test_datasets(features='mass,d2', train_size=100, test_size=900, seed=1, style='qiskit'):
    assert (train_size >
            0), f'train_size must be greater than 0, but is "{train_size}".'
    assert (
        test_size > 0), f'test_size must be greater than 0, but is "{test_size}".'
    assert style in [
        'qiskit', 'sklearn'], f'style must be "qiskit" or "sklearn" but was "{style}".'

    if features == 'all':
        features = validate.FEATURES
    else:
        features = utils.str_to_list(features)
    validate.features(features)

    n = train_size + test_size
    X_background = load_samples(gen_type='qcd', n=n//2)[features].to_numpy()
    X_signal = load_samples(gen_type='higgs', n=(n-n//2))[features].to_numpy()
    y_background = np.zeros(X_background.shape[0])
    y_signal = np.ones(X_signal.shape[0])

    X = np.concatenate([X_background, X_signal], axis=0)
    y = np.concatenate([y_background, y_signal])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed, train_size=train_size, test_size=test_size)

    if style == 'sklearn':
        return X_train, y_train, X_test, y_test

    train_dataset = {'qcd': X_train[y_train ==
                                    0], 'higgs': X_train[y_train == 1]}
    test_dataset = {'qcd': X_test[y_test == 0], 'higgs': X_test[y_test == 1]}

    return train_dataset, test_dataset
