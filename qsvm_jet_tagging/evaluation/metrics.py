import argparse
import logging
import os

from common import utils
from sklearn.externals import joblib


def load_run_config(run_path):

    # Load settings from the config.
    config = utils.load_run_config(run_path)
    ml_settings = config['ML Settings']
    model_name = ml_settings['model name']
    features = utils.str_to_list(ml_settings['features'])
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


def print_metrics(run_path):

    print(f'Printing metrics for run at {run_path}.')

    # Load settings from the config.
    model_name, features, train_size, test_size, seed = load_run_config(
        run_path)

    # Load the model and result from the run.
    model_path = os.path.join(run_path, 'model.joblib')
    result_path = os.path.join(run_path, 'result.joblib')
    model = joblib.load(model_path)
    result = joblib.load(result_path)

    # Print the testing accuracy of the model.
    testing_accuracy = result['testing_accuracy']
    print(f'\tTesting accuracy = {testing_accuracy}')


def main():
    parser = argparse.ArgumentParser(
        description='Display performance metrics for the given run.')
    parser.add_argument(
        '--run', help='The run number corresponding to the run that should be evaluated. By default, the most recent run is used.')

    args = parser.parse_args()

    if args.run is None:
        run_path = utils.most_recent_dir()
    else:
        run_path = os.path.join(utils.get_results_path(), args.run)

    print_metrics(run_path)


if __name__ == '__main__':
    main()
