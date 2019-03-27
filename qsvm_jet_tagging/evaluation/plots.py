"""Plotting

These routines handle creating various plots that are useful for evaluation.
"""

import argparse
import logging

import seaborn as sns
import sklearn
from common import utils
from matplotlib import pyplot as plt
from training import train

sns.set()


def plot_roc_curve(run_path):

    logging.info(f'Drawing ROC curve for run at {run_path}')

    # Load from the run.
    model_name, features, train_size, test_size, seed = utils.load_run_config_settings(
        run_path)
    model, result = utils.load_model(run_path)
    _, _, X_test, y_test = train.get_train_test_datasets(
        features, train_size, test_size, seed, style='sklearn')

    # Plot ROC curve.
    y_score = model.decision_function(X_test)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_score)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


def plot_confidence(run_path, use_training_data=False):

    # Load from the run.
    model_name, features, train_size, test_size, seed = utils.load_run_config_settings(
        run_path)
    model, result = utils.load_model(run_path)
    X_train, y_train, X_test, y_test = train.get_train_test_datasets(
        features, train_size, test_size, seed, style='sklearn')

    if use_training_data:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test

    # Get confidence for signal/background
    X_signal = X[y == 1]
    X_background = X[y == 0]
    confidence_signal = model.decision_function(X_signal)
    confidence_background = model.decision_function(X_background)

    # Make the two confidence plots.
    sns.distplot(confidence_signal, color='b', label='Higgs', kde=False)
    sns.distplot(confidence_background, color='r', label='QCD', kde=False)
    plt.legend()
    plt.title(f'{model_name} Confidences')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot the given curve for the given run.')
    parser.add_argument(
        'type', choices=['roc_curve', 'confidence'], help='The type of plot to make.')
    parser.add_argument(
        '--run', help='The run that should be plotted. By default, uses the most-recent run.')
    parser.add_argument('--use_training_data', action='store_true',
                        help='If set, plot given curve for training data (if supported). By default, testing data is used.')
    args = parser.parse_args()

    if args.run is None:
        run_path = utils.most_recent_run_path()
    else:
        run_path = utils.get_run_path(args.run)

    if args.type == 'roc_curve':
        plot_roc_curve(run_path)
    elif args.type == 'confidence':
        plot_confidence(run_path, args.use_training_data)


if __name__ == '__main__':
    main()
