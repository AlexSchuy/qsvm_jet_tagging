import argparse
import logging
import os

from common import persistence, utils
from sklearn.externals import joblib
from training.run import Run


def print_metrics(run):

    print(f'Printing metrics for run at {run.run_path}.')

    testing_accuracy = run.result['testing_accuracy']
    print(f'\tTesting accuracy = {testing_accuracy}')

    if run.model_name in ('qsvm_kernel', 'svm_classical'):
        support = run.result['svm']['support_vectors'].shape[0]
        print(f'\tsupport = {support}/{run.train_size}')
    elif run.model_name == 'sklearn_svm':
        support = len(run.model.best_estimator_.named_steps['svc'].support_)
        print(f'\tsupport = {support}/{run.train_size}')

    X_train, y_train, _, _ = run.get_train_test_datasets()

    if 'training_accuracy' not in run.result:
        training_accuracy = run.model.score(X_train, y_train)
        run.result['training_accuracy'] = training_accuracy
        run.save()
    else:
        training_accuracy = run.result['training_accuracy']
    print(f'\tTraining accuracy = {training_accuracy}')

    print(f'\tTraining time = {run.result["training_time"]}')
    print(f'\tTesting time = {run.result["testing_time"]}')


def main():
    parser = argparse.ArgumentParser(
        description='Display performance metrics for the given run.')
    parser.add_argument(
        '--run', help='The run number corresponding to the run that should be evaluated. By default, the most recent run is used.')

    args = parser.parse_args()

    if args.run is None:
        run = Run.most_recent()
    else:
        run = Run(args.run)

    print_metrics(run)


if __name__ == '__main__':
    main()
