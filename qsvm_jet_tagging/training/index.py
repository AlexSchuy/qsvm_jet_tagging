"""Contains methods to manipulate the index for runs (see training/run.py)."""


import functools
import operator
import os

import numpy as np
import pandas as pd
from common import utils


def validate_index(index, config):
    assert index.empty or set(index.columns.values) == set(config.keys(
    )), f'index contains columns {index.columns.values} but given config has keys {list(config.keys())}. Please regenerate the index.'


def get_index_path():
    return os.path.join(utils.get_results_path(), 'index.csv')


def load():
    index_path = get_index_path()
    if os.path.exists(index_path):
        index = pd.read_csv(index_path, index_col='run_number')
    else:
        index = pd.DataFrame()
    return index


def get_run_number(config):
    index = load()
    if index.empty:
        return -1
    try:
        matches_one = [index[k] == v for k, v in config.items()]
        matches_all = functools.reduce(operator.and_, matches_one)
        matches = index.index[matches_all]
        assert len(
            matches) <= 1, 'Index corrupted, multiple run numbers correspond to the given config. Please regenerate the index.'
        return matches[0]
    except IndexError:
        return -1


def get_config(run_number):
    index = load()
    if index.empty:
        return None
    try:
        entry = index.loc[run_number]
        config = dict(entry)
        return config
    except KeyError:
        return None


def add(run_number, config):
    assert get_run_number(config) == -1, f'{config} already present in index.'
    assert get_config(
        run_number) is None, f'{run_number} already present in index.'
    assert all(not type(v) == list for k, v in config.items()
               ), 'Lists cannot be stored as elements in the index.'
    index = load()
    validate_index(index, config)
    if index.empty:
        index = pd.DataFrame(columns=list(config.keys()))
    index.loc[run_number] = config
    index.to_csv(get_index_path(), index_label='run_number')


def remove(run_number, config):
    assert get_run_number(
        config) == run_number, f'({run_number}, {config}) is not present in index.'
    index = load()
    validate_index(index, config)
    index.drop(run_number, inplace=True)
    index.to_csv(get_index_path())
