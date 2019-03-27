import argparse
import logging
import os

from common import utils
from sklearn.externals import joblib


def print_metrics(run_path):

    print(f'Printing metrics for run at {run_path}.')

    # Load settings from the config.
    model_name, features, train_size, test_size, seed = utils.load_run_config_settings(
        run_path)

    # Load the model and result from the run.
    model, result = utils.load_model(run_path)

    # Print the testing accuracy of the model.
    testing_accuracy = result['testing_accuracy']
    if model_name in ('qsvm_kernel', 'svm_classical'):
        support = result['svm']['support_vectors'].shape[0]
    elif model_name == 'sklearn_svm':
        support = len(model.best_estimator_.named_steps['svc'].support_)
    print(f'\tTesting accuracy = {testing_accuracy}')
    print(f'\tsupport = {support}/{train_size}')


def main():
    parser = argparse.ArgumentParser(
        description='Display performance metrics for the given run.')
    parser.add_argument(
        '--run', help='The run number corresponding to the run that should be evaluated. By default, the most recent run is used.')

    args = parser.parse_args()

    if args.run is None:
        run_path = utils.most_recent_run_path()
    else:
        run_path = os.path.join(utils.get_results_path(), args.run)

    print_metrics(run_path)


if __name__ == '__main__':
    main()
