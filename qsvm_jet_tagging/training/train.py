"""Training

These routines handle training a ML model.
"""

import argparse
import logging
import os
from configparser import ConfigParser

import numpy as np
from common import utils, validate
from generation.generate_samples import load_samples
from qiskit import Aer
from qiskit_aqua import QuantumInstance, run_algorithm, set_aqua_logging
from qiskit_aqua.algorithms import QSVMKernel
from qiskit_aqua.components.feature_maps import SecondOrderExpansion
from qiskit_aqua.input import SVMInput
from qiskit_aqua.utils import (map_label_to_class_name,
                               split_dataset_to_data_and_labels)
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_train_test_datasets(features=['mass', 'd2'], train_size=100, test_size=900, seed=1, style='qiskit'):

    validate.features(features)
    assert (train_size >
            0), f'train_size must be greater than 0, but is "{train_size}".'
    assert (
        test_size > 0), f'test_size must be greater than 0, but is "{test_size}".'
    assert style in [
        'qiskit', 'sklearn'], f'style must be "qiskit" or "sklearn" but was "{style}".'

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


def train_qsvm_kernel(training_dataset, testing_dataset=None, seed=10598):
    feature_dim = 2
    feature_map = SecondOrderExpansion(
        num_qubits=feature_dim, depth=2, entanglement='linear')
    qsvm = QSVMKernel(feature_map, training_dataset,
                      test_dataset=testing_dataset)

    backend = Aer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(
        backend, shots=1024, seed=seed, seed_mapper=seed)
    result = qsvm.run(quantum_instance, print_info=True)

    return (qsvm, result)


def train_model(model_name, features=['mass', 'd2'], train_size=100, test_size=900, seed=10598):
    validate.model_name(model_name)
    validate.features(features)
    assert train_size > 0, f'train_size must be greater than 0, but is "{train_size}".'
    assert test_size > 0, f'test_size must be greater than 0, but is "{test_size}".'

    # Train the model.
    if model_name == 'qsvm_kernel':
        training_dataset, testing_dataset = get_train_test_datasets(
            features, train_size, test_size, seed=seed, style='qiskit')
        model, result = train_qsvm_kernel(
            training_dataset, testing_dataset, seed=seed)
    elif model_name == 'sklearn_svm':
        raise NotImplementedError()
    elif model_name == 'qsvm_variational':
        raise NotImplementedError()

    # Create a run dir.
    run_path = utils.make_run_dir()
    logging.info(f'Storing model in {run_path}.')

    # Save configuration settings in the run dir.
    config = ConfigParser()
    config['ML Settings'] = {}
    config['ML Settings']['model_name'] = model_name
    config['ML Settings']['features'] = utils.list_to_str(features)
    config['ML Settings']['train size'] = str(train_size)
    config['ML Settings']['test size'] = str(test_size)
    config['ML Settings']['seed'] = str(seed)
    utils.save_run_config(config, run_path)

    # Save the model in the run dir.
    model_path = os.path.join(run_path, 'model.joblib')
    joblib.dump(model, model_path)


def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument(
        'model_name', choices=validate.MODEL_NAMES, help='The model to train.')
    parser.add_argument('--features', '-f', required=True, nargs='+',
                        choices=validate.FEATURES, help='The features to train.')
    parser.add_argument('--train_size', type=int, default=5,
                        help='The number of training samples to use.')
    parser.add_argument('--test_size', type=int, default=95,
                        help='The number of testing samples to use.')
    parser.add_argument('--seed', type=int, default=10598,
                        help='The random seed to use.')

    args = parser.parse_args()

    train_model(args.model_name, args.features,
                args.train_size, args.test_size, args.seed)


if __name__ == '__main__':
    main()
