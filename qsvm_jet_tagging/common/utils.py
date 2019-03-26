"""Utility Routines

These routines handle common utility functions for the other modules, such as
get common paths and handling config files.
"""

import os
from configparser import ConfigParser


def get_source_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_path():
    return os.path.dirname(get_source_path())


def get_results_path():
    return os.path.join(get_project_path(), 'results')


def load_run_config(run_path):
    config = ConfigParser(allow_no_value=True)
    config.read(os.path.join(run_path, 'run.ini'))
    return config


def save_run_config(config, run_path):
    with open(os.path.join(run_path, 'run.ini'), 'w') as config_file:
        config.write(config_file)


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


def most_recent_dir():
    results_path = get_results_path()
    return max((os.path.join(results_path, d) for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))), key=lambda f: int(os.path.basename(f)), default=None)
