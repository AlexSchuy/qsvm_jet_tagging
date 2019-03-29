"""Implements a class that encapsulates the notion of a single 'run' over the
data using specific settings.
"""
import argparse
import os
from configparser import ConfigParser

import numpy as np
from common import utils, validate
from sklearn.externals import joblib
from training import data, index, train


class Run():
    """Stores the results of a single run over the data with specific settings. 

    Runs are saved to disk and indexed. A Run can be loaded from
    either a run number or a config. If loaded from a config, a new run will
    be performed if one is not already present in the index. In this way,
    it is possible to load a run with specific settings and not worry about
    whether a model with those settings has already been trained, while still 
    avoiding duplication.
    """

    def __init__(self, identifier):
        """Load from the given identifier. If identifier is a dict, the run index will be checked to see whether a run with those 
        settings has been performed. If not, a new run will be performed.

        Parameters
        ----------
        identifier : {int, dict}
            If int, the number of a run.
            If dict, a config detailing the run.
        """
        if type(identifier) == int:
            self._load_from_run_number(identifier)
        elif type(identifier) == dict:
            self._load_from_config(identifier)

    @classmethod
    def most_recent(cls):
        run_number = cls._most_recent_run_number()
        return cls(run_number)

    def get_train_test_datasets(self, style='sklearn'):
        return data.get_train_test_datasets(self.features, self.train_size, self.test_size, self.seed, style)

    @property
    def run_path(self):
        return self._get_run_path(self.run_number)

    @property
    def model(self):
        return self._model

    @property
    def result(self):
        return self._result

    @staticmethod
    def _get_run_path(run_number):
        return os.path.join(utils.get_results_path(), format(run_number, '03d'))

    @classmethod
    def _most_recent_run_number(cls):
        results_path = utils.get_results_path()
        return max((int(d) for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))), default=-1)

    @classmethod
    def _most_recent_run_path(cls):
        results_path = utils.get_results_path()
        run_number = cls._most_recent_run_number()
        if run_number == -1:
            return None
        else:
            return cls._get_run_path(run_number)

    def _load_model(self):
        # This is an ugly hack that gets around qsvm_variational pickling
        # issues.
        if self.model_name == 'qsvm_variational':
            model_path = os.path.join(self.run_path, 'model.npz')
            self._model = QSVMVariationalClassifier(seed=seed)
            X, y, _, _ = get_train_test_datasets(
                self.features, self.train_size, self.test_size, self.seed, style='sklearn')
            self._model.load_model(model_path, X, y)
        else:
            model_path = os.path.join(self.run_path, 'model.joblib')
            self._model = joblib.load(model_path)

        result_path = os.path.join(self.run_path, 'result.joblib')
        self._result = joblib.load(result_path)

    def _save_model(self):
        if self.model_name == 'qsvm_variational':
            model_path = os.path.join(self.run_path, 'model.npz')
            self._model.save_model(model_path)
        else:
            model_path = os.path.join(self.run_path, 'model.joblib')
            joblib.dump(self._model, model_path)

        result_path = os.path.join(self.run_path, 'result.joblib')
        joblib.dump(self._result, result_path)

    def _set_config(self, config):
        for k, v in config.items():
            if type(v) == list:
                config[k] = utils.list_to_str(v)
        train.add_default_settings(config)
        self.config = config
        for k, v in self.config.items():
            setattr(self, str(k), v)

    def _load_from_run_number(self, run_number):
        self.run_number = run_number
        config = index.get_config(run_number)
        if config is None:
            raise IndexError(f'run {run_number} is not present in the index.')
        self._set_config(config)
        self._load_model()

    def _load_from_config(self, config):
        self._set_config(config.copy())
        run_number = index.get_run_number(self.config)
        if run_number != -1:
            self._load_from_run_number(run_number)
        else:
            # The model isn't in the index, so train it and add it.
            self._model, self._result = train.train_model(**self.config)
            self.run_number = self._most_recent_run_number() + 1
            os.makedirs(self.run_path)
            self._save_model()
            index.add(self.run_number, self.config)


def main():
    parser = argparse.ArgumentParser(
        description='Make a run with the given settings.')
    parser.add_argument(
        'model_name', choices=validate.MODEL_NAMES, help='The model to train.')
    parser.add_argument('--features', '-f', required=True, nargs='+',
                        choices=validate.FEATURE_CHOICES, help='A list of the features to train, or "all" for all features.')
    parser.add_argument(
        '--pca', type=int, help='If given, use principal component analysis to reduce the features down to the given number of dimensions.', default=0)
    parser.add_argument('--train_size', type=int, default=5,
                        help='The number of training samples to use.')
    parser.add_argument('--test_size', type=int, default=95,
                        help='The number of testing samples to use.')
    parser.add_argument('--seed', type=int, default=10598,
                        help='The random seed to use.')

    args = parser.parse_args()

    config = {'model_name': args.model_name, 'features': utils.list_to_str(
        args.features), 'train_size': args.train_size, 'test_size': args.test_size, 'seed': args.seed, 'pca': args.pca}

    run = Run(config)


if __name__ == '__main__':
    main()
