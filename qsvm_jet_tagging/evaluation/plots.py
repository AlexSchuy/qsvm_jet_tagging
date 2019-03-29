"""Plotting

These routines handle creating various plots that are useful for evaluation.
"""
import argparse
import logging
import os

import seaborn as sns
import sklearn
from common import persistence, utils
from matplotlib import pyplot as plt
from training import train
from training.run import Run

sns.set()


def plot_roc_curve(run):

    if 'roc_curve_plot' in run.result:
        img = plt.imread(run.result['roc_curve_plot'])
        plt.imshow(img)
    else:
        _, _, X_test, y_test = run.get_train_test_datasets()

        y_score = run.model.decision_function(X_test)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_score)
        plt.title('ROC')
        plt.plot(fpr, tpr, 'b')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Signal Efficiency')
        plt.xlabel('Background Efficiency')
        plot_path = os.path.join(run.run_path, 'roc_curve.png')
        plt.savefig(plot_path)
        print(f'Saved roc curve plot at {plot_path}.')
        run.result['roc_curve_plot'] = plot_path
        run.save()
        plt.show()


def plot_scores(run, use_training_data=False):

    if 'scores_plot' in run.result:
        img = plt.imread(run.result['scores_plot'])
        plt.imshow(img)
    else:
        X_train, y_train, X_test, y_test = run.get_train_test_datasets()

        if use_training_data:
            X = X_train
            y = y_train
        else:
            X = X_test
            y = y_test

        # Get scores for signal/background
        X_signal = X[y == 1]
        X_background = X[y == 0]
        scores_signal = run.model.decision_function(X_signal)
        scores_background = run.model.decision_function(X_background)

        # Make the two confidence plots.
        sns.distplot(scores_signal, color='b', label='Higgs', kde=False)
        sns.distplot(scores_background, color='r', label='QCD', kde=False)
        plt.legend()
        plt.title('Scores')
        plot_path = os.path.join(run.run_path, 'scores.png')
        plt.savefig(plot_path)
        print(f'Saved score plot at {plot_path}.')
        run.result['scores_plot'] = plot_path
        run.save()
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
        run = Run.most_recent()
    else:
        run_path = Run(args.run)

    if args.type == 'roc_curve':
        plot_roc_curve(run)
    elif args.type == 'scores':
        plot_scores(run, args.use_training_data)


if __name__ == '__main__':
    main()
