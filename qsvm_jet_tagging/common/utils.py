"""Utility Routines

These routines handle common utility functions for the other modules, such as
get common paths and handling config files.
"""
import os
from configparser import ConfigParser

from sklearn.externals import joblib
from training import qsvm_kernel
from training.qsvm_variational import QSVMVariationalClassifier
from training.train import get_train_test_datasets


def get_source_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_path():
    return os.path.dirname(get_source_path())


def get_results_path():
    return os.path.join(get_project_path(), 'results')


def get_run_path(run_number):
    return os.path.join(get_results_path(), run_number)


def load_run_config(run_path):
    config = ConfigParser(allow_no_value=True)
    config.read(os.path.join(run_path, 'run.ini'))
    return config


def load_run_config_settings(run_path):
    # Load settings from the config.
    config = load_run_config(run_path)
    ml_settings = config['ML Settings']
    model_name = ml_settings['model name']
    features = str_to_list(ml_settings['features'])
    train_size = int(ml_settings['train size'])
    test_size = int(ml_settings['test size'])
    seed = int(ml_settings['seed'])

    print('The run had the following settings:')
    print(f'\tmodel_name={model_name}')
    print(f'\tfeatures={features}')
    print(f'\ttrain_size={train_size}')
    print(f'\ttest_size={test_size}')
    print(f'\tseed={seed}')

    return model_name, features, train_size, test_size, seed


def save_run_config(config, run_path):
    with open(os.path.join(run_path, 'run.ini'), 'w') as config_file:
        config.write(config_file)


def load_model(run_path):
    result_path = os.path.join(run_path, 'result.joblib')
    model_name, features, train_size, test_size, seed = load_run_config_settings(
        run_path)
    if model_name == 'qsvm_variational':
        model_path = os.path.join(run_path, 'model.npz')
        model = QSVMVariationalClassifier(seed=seed)
        X, y, _, _ = get_train_test_datasets(
            features, train_size, test_size, seed, style='sklearn')
        model.load_model(model_path, X, y)
    else:
        model_path = os.path.join(run_path, 'model.joblib')
        model = joblib.load(model_path)
    result = joblib.load(result_path)

    return (model, result)


def save_model(model_name, model, result, run_path):
    result_path = os.path.join(run_path, 'result.joblib')
    if model_name == 'qsvm_variational':
        model_path = os.path.join(run_path, 'model.npz')
        model.save_model(model_path)
    else:
        model_path = os.path.join(run_path, 'model.joblib')
        joblib.dump(model, model_path)
    joblib.dump(result, result_path)


def list_to_str(l):
    s = ''
    for item in l[:-1]:
        s += str(item) + ','
    s += l[-1]
    return s


def str_to_list(s, type_func=str):
    return [type_func(i) for i in s.split(',')]


def make_run_dir():
    results_path = get_results_path()
    os.makedirs(results_path, exist_ok=True)
    run_number = most_recent_run_number() + 1
    run_path = os.path.join(results_path, format(run_number, '03d'))
    os.makedirs(run_path)
    config = ConfigParser()
    with open(os.path.join(run_path, 'run.ini'), 'w') as config_file:
        config.write(config_file)
    return run_path


def most_recent_run_number():
    results_path = get_results_path()
    return max((int(d) for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))), default=-1)


def most_recent_run_path():
    results_path = get_results_path()
    return max((os.path.join(results_path, d) for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))), key=lambda f: int(os.path.basename(f)), default=None)


def make_qiskit_style_dataset(X, y):
    return {'0': X[y == 0], '1': X[y == 1]}
