"""Input Validation

These routines handle common input validation.
"""


FEATURES = ('pt', 'eta', 'phi', 'mass', 'ee2', 'ee3', 'd2')
MODEL_NAMES = ('qsvm_kernel', 'qsvm_variational',
               'svm_classical', 'sklearn_svm')


def model_name(model_name):
    assert model_name in MODEL_NAMES, f'"{model_name}" is an invalid model name. Valid model names are {MODEL_NAMES}.'


def features(features):
    assert all(
        f in FEATURES for f in features), f'{features} contains an invalid feature. Valid features are: {FEATURES}'
    assert len(
        features) == 2, f'There must be exactly 2 features, but features={features}'
