import os

import numpy as np
from common import utils, validate
from generation.generate_samples import load_samples
from sklearn.datasets import load_breast_cancer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_train_test_datasets(features='mass,d2', train_size=100, test_size=900, seed=10598, style='qiskit', dataset='higgs_tagging'):
    assert (train_size >
            0), f'train_size must be greater than 0, but is "{train_size}".'
    assert (
        test_size > 0), f'test_size must be greater than 0, but is "{test_size}".'
    assert style in (
        'qiskit', 'sklearn'), f'style must be "qiskit" or "sklearn" but was "{style}".'
    assert dataset in (
        'higgs_tagging', 'breast_cancer'), f'dataset must be "higgs_tagging" or "breast_cancer" but was "{dataset}".'

    if dataset == 'higgs_tagging':
        labels = {0: 'qcd', 1: 'higgs'}

        if features == 'all':
            features = validate.FEATURES
        else:
            features = utils.str_to_list(features)
        validate.features(features)

        n = train_size + test_size
        X_background = load_samples(
            gen_type='qcd', n=n//2)[features].to_numpy()
        X_signal = load_samples(
            gen_type='higgs', n=(n-n//2))[features].to_numpy()
        y_background = np.zeros(X_background.shape[0])
        y_signal = np.ones(X_signal.shape[0])

        X = np.concatenate([X_background, X_signal], axis=0)
        y = np.concatenate([y_background, y_signal])
    elif dataset == 'breast_cancer':
        labels = {0: 'malignant', 1: 'benign'}
        X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed, train_size=train_size, test_size=test_size)

    if style == 'sklearn':
        return X_train, y_train, X_test, y_test

    train_dataset = {labels[0]: X_train[y_train ==
                                        0], labels[1]: X_train[y_train == 1]}
    test_dataset = {labels[0]: X_test[y_test == 0],
                    labels[1]: X_test[y_test == 1]}

    return train_dataset, test_dataset
