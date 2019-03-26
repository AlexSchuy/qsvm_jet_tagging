import sklearn
from generation import generate_samples


def get_train_test_datasets(features=['mass', 'd2'], train_size=100, test_size=900, seed=1, style='qiskit'):
    assert len(
        features) == 2, f'There must be exactly 2 features, but features={features}'
    assert train_size > 0, f'train_size={train_size} must be at least 1.'
    assert test_size > 0, f'test_size={test_size} must be at least 1.'
    assert style in [
        'qiskit', 'sklearn'], f'style must be "qiskit" or "sklearn" but was "{style}".'

    n = train_size + test_size
    background = generate_samples(gen_type='qcd', n=n//2)[features].to_numpy()
    signal = generate_samples(gen_type='higgs', n=n//2)[features].to_numpy()
    y_background = np.zeros(background.shape[0])
    y_signal = np.ones(signal.shape[0])

    X = np.concatenate(X_background, X_signal)
    y = np.concatenate(y_background, y_signal)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, stratify=y, random_state=seed)

    if style == 'sklearn':
        return X_train, y_train, X_test, y_test

    train_dataset = {'qcd': X_train[y_train ==
                                    0], 'higgs': X_train[y_train == 1]}
    test_dataset = {'qcd': X_test[y_test == 0], 'higgs': X_test[y_test == 1]}

    return train_dataset, test_dataset


def train_qsvm_kernel(training_dataset):
    seed = 10598
    feature_dim = 2
    feature_map = SecondOrderExpansion(
        num_qubits=feature_dim, depth=2, entanglement='linear')
    qsvm = QSVMKernel(feature_map, training_dataset)
