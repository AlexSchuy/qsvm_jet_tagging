"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""

import argparse
import logging
import os
import time
from configparser import ConfigParser

import numpy as np
from common import persistence, utils, validate
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from training.data import get_train_test_datasets
from training.qsvm_kernel import QSVMKernelClassifier
from training.qsvm_variational import QSVMVariationalClassifier
from training.svm_classical import SVMClassicalClassifier


def make_pipeline(model, model_scaler, pca, seed):
    model_steps = [('model_scaler', model_scaler), ('model', model)]
    if pca == 0:
        pipeline = Pipeline(steps=model_steps)
    else:
        pca_steps = [('pca_scaler', StandardScaler()),
                     ('pca', PCA(pca, random_state=seed))]
        pca_steps.extend(model_steps)
        pipeline = Pipeline(steps=pca_steps)
    return pipeline


def make_svm_classical(gamma=None, seed=10598, pca=0):
    svm = SVMClassicalClassifier(gamma)
    return make_pipeline(svm, StandardScaler(), pca, seed)


def make_qsvm_kernel(seed=10598, pca=0):
    qsvm = QSVMKernelClassifier(seed)
    return make_pipeline(qsvm, MinMaxScaler((-1, 1)), pca, seed)


def make_qsvm_variational(seed=10598, pca=0):
    qsvm = QSVMVariationalClassifier(seed)
    return make_pipeline(qsvm, MinMaxScaler((-1, 1)), pca, seed)


def make_sklearn_svm(seed=10598, pca=0):

    # Use a default SVM.
    svc = svm.SVC()
    pipeline = make_pipeline(svc, StandardScaler(), pca, seed)

    # Apply a grid search to tune hyperparameters.
    C_choices = [2**p for p in range(-5, 15, 2)]
    gamma_choices = [2**p for p in range(-15, 3, 2)]
    poly_grid = {'model__kernel': ['poly'], 'model__degree': [
        2, 3], 'model__gamma': gamma_choices, 'model__C': C_choices}
    rbf_grid = {'model__kernel': ['rbf'],
                'model__gamma': gamma_choices, 'model__C': C_choices}
    param_grid = [poly_grid, rbf_grid]
    cv = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=5, iid=False)

    return cv


def make_model(model_name, seed, pca):
    validate.model_name(model_name)
    if model_name == 'sklearn_svm':
        return make_sklearn_svm(seed=seed, pca=pca)
    elif model_name == 'qsvm_kernel':
        return make_qsvm_kernel(seed=seed, pca=pca)
    elif model_name == 'qsvm_variational':
        return make_qsvm_variational(seed=seed, pca=pca)
    elif model_name == 'svm_classical':
        return make_svm_classical(seed=seed, pca=pca)
    else:
        raise NotImplementedError()


def train_model(model_name, features, train_size, test_size, seed, pca):
    validate.model_name(model_name)
    assert train_size > 0, f'train_size must be greater than 0, but is "{train_size}".'
    assert test_size > 0, f'test_size must be greater than 0, but is "{test_size}".'

    # Train the model.
    X_train, y_train, X_test, y_test = get_train_test_datasets(
        features, train_size, test_size, seed=seed, style='sklearn')
    model = make_model(model_name, seed=seed, pca=pca)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    print(f'Total training time: {training_time} s')

    # Get results.
    if model_name == 'sklearn_svm':
        # Create a 'result' dict similar to qiskit.
        result = {}
    else:
        result = model.named_steps['model'].ret()

    # Add testing accuracy to the results for convenience.
    start = time.time()
    result['testing_accuracy'] = model.score(X_test, y_test)
    testing_time = time.time() - start
    print(f'Total testing time: {testing_time} s')

    # Add timing to results.
    result['training_time'] = training_time
    result['testing_time'] = testing_time

    return (model, result)


def add_default_settings(config):
    defaults = {'features': 'mass,d2', 'train_size': 100,
                'test_size': 100, 'seed': 10598, 'pca': 0}
    for k, v in defaults.items():
        if k not in config:
            config[k] = v
